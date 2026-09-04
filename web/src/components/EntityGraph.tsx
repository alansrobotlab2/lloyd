import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import SpriteText from "three-spritetext";
import { forceCollide } from "d3-force";
import { api, type EntityGraphNode, type EntityGraphData } from "../api";
import { RotateCcw, Maximize } from "lucide-react";
import { categoryOf, CATEGORY_COLOR } from "../lib/edgeCategories";
import { KIND_COLOR, KIND_ORDER } from "../lib/entityKinds";

// -- Types --

interface GNode extends EntityGraphNode {
  // d3 simulation props (added at runtime)
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
  fx?: number;
  fy?: number;
  fz?: number;
  // computed
  degree?: number;
}

interface GLink {
  source: string | GNode;
  target: string | GNode;
  type: string;
  weight: number;
}

interface GData {
  nodes: GNode[];
  links: GLink[];
}

export interface EntityGraphProps {
  /** External selection — node id to highlight */
  selectedNode?: string | null;
  /** Called when a node is single-clicked */
  onNodeClick?: (nodeId: string | null) => void;
  /** Called when a node is double-clicked */
  onNodeDoubleClick?: (nodeId: string) => void;
  /** Called when a node is hovered (debounced ~150ms). null = no hover */
  onNodeHover?: (node: GNode | null) => void;
  /** Surface the loaded graph payload to the parent so it can be reused
   *  (connection counts, side-panel rendering) without a duplicate fetch. */
  onGraphLoaded?: (data: EntityGraphData) => void;
}

// Edge filter state — one toggle per EdgeCategory defined in lib/edgeCategories.
interface EdgeFilters {
  structural: boolean;
  lineage: boolean;
  comparison: boolean;
  reference: boolean;
}

// -- Helpers --

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// Node colour is keyed on the entity's KIND (app/entity_kind.py), which is
// what the legend below names. It used to be keyed on `type === "entity"`
// with a fallback that hashed the node id into a hue — so every node that
// was not literally typed "entity" got an arbitrary colour and the legend
// described nothing.
//
// The palette itself lives in lib/entityKinds so MemoryPage can read it
// without importing this module (and with it three.js). Re-exported here for
// anything already importing it from the component.
export { KIND_COLOR, KIND_ORDER };

function nodeColor(node: GNode): string {
  return KIND_COLOR[node.type] || KIND_COLOR.entity;
}

function nodeRadius(node: GNode): number {
  const degree = node.degree ?? 0;
  return Math.max(2.5, Math.min(3.5, 2.5 + Math.sqrt(degree) * 0.15));
}

// THREE.Color can't parse the alpha component of "rgba(...)" strings — it
// warns and drops the alpha. Split it ourselves so we can feed Color a
// color-only string and apply the alpha via material.opacity.
function splitRgba(rgba: string): { color: string; alpha: number } {
  const m = rgba.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/);
  if (!m) return { color: rgba, alpha: 1 };
  return {
    color: `rgb(${m[1]}, ${m[2]}, ${m[3]})`,
    alpha: m[4] !== undefined ? parseFloat(m[4]) : 1,
  };
}

function edgeColor(type: string, highlighted: boolean, dimmed: boolean): string {
  if (dimmed) return "rgba(71,85,105,0.08)";
  const rgb = CATEGORY_COLOR[categoryOf(type)];
  // Highlighted: strong alpha so the selected subgraph pops.
  // Default (no selection): faint enough that highlighted edges still stand
  // out, but visible enough to read the topology.
  const alpha = highlighted ? 0.9 : 0.18;
  return `rgba(${rgb},${alpha})`;
}

// -- Shared node geometry --
//
// One geometry per node SHAPE, scaled per node, instead of a fresh
// SphereGeometry(r, 20, 16) for each of ~2,500 nodes — that cost ~215ms of
// every mount, and at the 3-4px a node occupies on screen the extra segments
// are detail nobody can see.
//
// three-forcegraph deallocates a node's object when the node leaves the graph
// (toggling an edge filter does exactly that), and _deallocate recursively
// calls geometry.dispose(). On a shared geometry that would free the buffer
// out from under every other node still using it, so these two — which live
// for the lifetime of the module — refuse to be disposed.
const SHARED_SPHERE = new THREE.SphereGeometry(1, 8, 6);
const SHARED_OCTAHEDRON = new THREE.OctahedronGeometry(1, 0);
SHARED_SPHERE.dispose = () => {};
SHARED_OCTAHEDRON.dispose = () => {};

/** Build a node's label sprite the first time it actually has to be shown.
 *
 *  SpriteText rasterises its text into a private canvas and uploads it as a
 *  texture. Doing that eagerly for every node cost ~1.3s of the mount plus
 *  ~2,500 texture uploads on the first frame — for labels that stay hidden
 *  until the node is highlighted or the camera comes close, which in practice
 *  is a few dozen of them. */
function ensureLabel(group: THREE.Object3D): THREE.Sprite {
  const cached = (group as any).__nodeLabel as THREE.Sprite | undefined;
  if (cached) return cached;

  const sprite = new SpriteText((group as any).__labelText ?? "");
  sprite.textHeight = 2.2;
  sprite.color = (group as any).__labelColor ?? KIND_COLOR.entity;
  sprite.backgroundColor = "rgba(15,23,42,0.65)";
  sprite.padding = 1.5;
  sprite.borderRadius = 2;
  sprite.position.set(0, (group as any).__labelOffset ?? 3, 0);
  // Don't intercept pointer events targeting the node behind the label.
  (sprite as any).raycast = () => {};

  group.add(sprite);
  (group as any).__nodeLabel = sprite;
  return sprite;
}

// -- Layout cache --
//
// A settled layout is worth keeping: recomputing it costs ~1.4s of blocking
// simulation on a graph this size, and the result is deterministic enough
// that reusing it is strictly better than watching the graph re-settle.
//
// Two tiers. The module-level cache survives a MemoryPage unmount (Layout
// renders one page at a time, so leaving the tab and coming back remounts
// this component from scratch); localStorage survives a reload.

const LAYOUT_STORAGE_KEY = "lloyd.entityGraph.layout.v1";
/** Refetch the graph if the module cache is older than this. */
const GRAPH_CACHE_TTL_MS = 5 * 60 * 1000;

interface GraphCache {
  fetchedAt: number;
  data: EntityGraphData;
  nodes: GNode[];
  links: GLink[];
}

/** Survives unmount, not reload. Holds the *same* node objects the simulation
 *  ran on, so their x/y/z and fx/fy/fz pins come back with them. */
let graphCache: GraphCache | null = null;

/** Order-independent digest of the node set, so a stored layout is only ever
 *  reapplied to the graph it was computed from. */
function layoutSignature(nodes: { id: string }[]): string {
  let sum = 0;
  for (const n of nodes) {
    let h = 0;
    for (let i = 0; i < n.id.length; i++) h = (h * 31 + n.id.charCodeAt(i)) | 0;
    sum = (sum + h) | 0;
  }
  return `${nodes.length}:${sum}`;
}

type StoredPositions = Record<string, [number, number, number]>;

function readStoredLayout(sig: string): StoredPositions | null {
  try {
    const raw = localStorage.getItem(LAYOUT_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { sig?: string; pos?: StoredPositions };
    return parsed.sig === sig && parsed.pos ? parsed.pos : null;
  } catch {
    return null;
  }
}

function writeStoredLayout(sig: string, nodes: GNode[]): void {
  try {
    const pos: StoredPositions = {};
    const r1 = (v: number) => Math.round(v * 10) / 10;
    for (const n of nodes) {
      if (typeof n.x !== "number" || typeof n.y !== "number") continue;
      pos[n.id] = [r1(n.x), r1(n.y), r1(n.z ?? 0)];
    }
    localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify({ sig, pos }));
  } catch {
    // Quota exceeded, or storage unavailable (private mode). The layout just
    // gets recomputed next reload — not worth failing the render over.
  }
}

/** Pin nodes at their stored positions. Returns true if the whole node set was
 *  covered, which is the condition for skipping warmup/cooldown entirely. */
function applyStoredLayout(nodes: GNode[], pos: StoredPositions): boolean {
  let applied = 0;
  for (const n of nodes) {
    const p = pos[n.id];
    if (!p) continue;
    n.x = n.fx = p[0];
    n.y = n.fy = p[1];
    n.z = n.fz = p[2];
    applied++;
  }
  return applied === nodes.length;
}

// -- Component --

export default function EntityGraph({ selectedNode: selectedNodeId, onNodeClick, onNodeDoubleClick, onNodeHover, onGraphLoaded }: EntityGraphProps) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const initialFitDone = useRef(false);
  const animFrameRef = useRef<number>(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rawGraphData, setRawGraphData] = useState<GData>({ nodes: [], links: [] });
  /** True once every node has a position to start from (module cache or
   *  localStorage), which lets the sim skip warmup and cooldown outright. */
  const [layoutPresettled, setLayoutPresettled] = useState(false);
  const layoutSigRef = useRef<string>("");
  const [activeNode, setActiveNode] = useState<GNode | null>(null);
  const [hoverNode, setHoverNode] = useState<GNode | null>(null);
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });
  /** The page is mounted but hidden (Layout keeps it around across tabs). */
  const [offscreen, setOffscreen] = useState(false);
  const [edgeFilters, setEdgeFilters] = useState<EdgeFilters>({
    structural: true,
    lineage: true,
    comparison: true,
    reference: true,
  });

  // Double-click detection
  const lastClickRef = useRef<{ nodeId: string; ts: number } | null>(null);

  // Hover debounce
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // -- Load data --

  const loadGraph = useCallback(async (force = false) => {
    // Remounting the page shouldn't re-pay for the fetch, the degree pass, or
    // the layout. The cached node objects are the ones d3 already positioned
    // and pinned, so handing them straight back gives a settled graph on the
    // first frame.
    if (!force && graphCache && Date.now() - graphCache.fetchedAt < GRAPH_CACHE_TTL_MS) {
      layoutSigRef.current = layoutSignature(graphCache.nodes);
      setLayoutPresettled(graphCache.nodes.every((n) => typeof n.fx === "number"));
      setRawGraphData({ nodes: graphCache.nodes, links: graphCache.links });
      onGraphLoaded?.(graphCache.data);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const data = await api.entityGraph();
      // Calculate degree for each node
      const degreeMap = new Map<string, number>();
      for (const edge of data.edges) {
        const src = typeof edge.source === "string" ? edge.source : (edge.source as any).id;
        const tgt = typeof edge.target === "string" ? edge.target : (edge.target as any).id;
        degreeMap.set(src, (degreeMap.get(src) || 0) + 1);
        degreeMap.set(tgt, (degreeMap.get(tgt) || 0) + 1);
      }
      const nodes: GNode[] = (data.nodes as GNode[]).map((n) => ({
        ...n,
        degree: degreeMap.get(n.id) || 0,
      }));
      const links = data.edges as GLink[];

      const sig = layoutSignature(nodes);
      layoutSigRef.current = sig;
      const stored = readStoredLayout(sig);
      setLayoutPresettled(stored ? applyStoredLayout(nodes, stored) : false);

      graphCache = { fetchedAt: Date.now(), data, nodes, links };
      setRawGraphData({ nodes, links });
      onGraphLoaded?.(data);
    } catch (err: any) {
      setError(err.message || "Failed to load entity graph");
    } finally {
      setLoading(false);
    }
  }, [onGraphLoaded]);

  useEffect(() => { loadGraph(); }, [loadGraph]);

  // -- Filtered graph data (apply edge type filters) --

  const graphData = useMemo((): GData => {
    const filteredLinks = rawGraphData.links.filter((l) => {
      const cat = categoryOf(l.type);
      return edgeFilters[cat] !== false;
    });

    // Remove orphaned nodes (nodes with no remaining edges)
    const connectedIds = new Set<string>();
    for (const l of filteredLinks) {
      const src = typeof l.source === "string" ? l.source : (l.source as GNode).id;
      const tgt = typeof l.target === "string" ? l.target : (l.target as GNode).id;
      connectedIds.add(src);
      connectedIds.add(tgt);
    }

    // Always keep the currently selected node to avoid UI glitch
    const filteredNodes = rawGraphData.nodes.filter(
      (n) => connectedIds.has(n.id) || n.id === selectedNodeId || n.id === activeNode?.id
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [rawGraphData, edgeFilters, selectedNodeId, activeNode]);

  // -- Highlight sets --

  const { highlightNodes, highlightLinks } = useMemo(() => {
    const hn = new Set<string>();
    const hl = new Set<string>();
    const focusId = activeNode?.id ?? null;
    if (focusId) {
      hn.add(focusId);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as GNode).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as GNode).id;
        if (src === focusId) { hn.add(tgt); hl.add(`${src}\x00${tgt}`); }
        if (tgt === focusId) { hn.add(src); hl.add(`${src}\x00${tgt}`); }
      }
    }
    if (hoverNode && !activeNode) {
      hn.add(hoverNode.id);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as GNode).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as GNode).id;
        if (src === hoverNode.id) { hn.add(tgt); hl.add(`${src}\x00${tgt}`); }
        if (tgt === hoverNode.id) { hn.add(src); hl.add(`${src}\x00${tgt}`); }
      }
    }
    return { highlightNodes: hn, highlightLinks: hl };
  }, [activeNode, hoverNode, graphData]);

  const hasDimming = !!(activeNode || hoverNode);

  // Which kinds are actually on screen, so the legend lists those and not
  // all eight.
  const kindsPresent = useMemo(
    () => new Set(graphData.nodes.map((n) => n.type)),
    [graphData],
  );

  // -- Resize tracking --
  //
  // Layout keeps the memory page mounted and hides it with `display:none`, so
  // a zero-sized observation means "the tab is in the background", not "the
  // graph shrank". Hold the last real size (resizing the canvas to 0 and back
  // costs a full WebGL context resize) and use it to park the render loop.

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const apply = (w: number, h: number) => {
      if (w > 0 && h > 0) {
        setDimensions({ w, h });
        setOffscreen(false);
      } else {
        setOffscreen(true);
      }
    };
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      apply(rect.width, rect.height);
    });
    ro.observe(container);
    const rect = container.getBoundingClientRect();
    apply(rect.width, rect.height);
    return () => ro.disconnect();
  }, []);

  // Stop rendering (and stop the label rAF loop below) while the tab is in the
  // background. Without this a hidden graph keeps drawing ~2,500 objects at
  // 60fps for as long as MC is open.
  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;
    if (offscreen) fg.pauseAnimation?.();
    else fg.resumeAnimation?.();
  }, [offscreen, loading]);

  // -- Fit to view once after first data load --

  useEffect(() => {
    if (!graphData.nodes.length || initialFitDone.current) return;
    const t = setTimeout(() => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(1000, 50);
        initialFitDone.current = true;
      }
    }, 400);
    return () => clearTimeout(t);
  }, [graphData]);

  // -- Force configuration --

  const configureForces = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    try {
      // iterations(1) rather than 2: collide is the most expensive force in
      // the set, and the second relaxation pass cost ~250ms per 60 ticks for
      // a separation nobody can see at this zoom.
      fg.d3Force("collide",
        forceCollide().radius(() => 8).strength(0.6).iterations(1)
      );
      fg.d3Force("charge")?.strength(-90).distanceMax(400);
      fg.d3Force("link")?.distance(() => 45).strength(0.3);
    } catch (e) {
      console.warn("configureForces failed:", e);
    }
  }, []);

  useEffect(() => {
    if (!rawGraphData.nodes.length) return;
    // Apply custom forces synchronously — BEFORE the library's debounced
    // updateFn runs. updateFn reads d3ForceLayout at the moment it fires and
    // then runs warmupTicks of the sim; if our forces are already in place
    // the sim warms up with them and we avoid the visible "re-settle" when
    // they're applied mid-animation. fg.d3Force() writes to state.d3ForceLayout
    // (always present from stateInit) so this is safe to call immediately.
    configureForces();
  }, [rawGraphData, configureForces]);

  // -- Sync with external selectedNode prop --

  useEffect(() => {
    if (!graphData.nodes.length || loading) return;
    if (!selectedNodeId) {
      if (activeNode) {
        setActiveNode(null);
        cancelAnimationFrame(animFrameRef.current);
      }
      return;
    }
    const nd = graphData.nodes.find((n) => n.id === selectedNodeId);
    if (nd && nd.id !== activeNode?.id) {
      setActiveNode(nd);
      if (fgRef.current && nd.x != null && nd.y != null) {
        const nx = nd.x, ny = nd.y, nz = nd.z ?? 0;
        const dist = 120;
        const mag = Math.hypot(nx, ny, nz) || 1;
        const k = 1 + dist / mag;
        fgRef.current.cameraPosition(
          { x: nx * k, y: ny * k, z: nz * k },
          { x: nx, y: ny, z: nz },
          800
        );
      }
    }
  }, [selectedNodeId, graphData, loading]); // eslint-disable-line react-hooks/exhaustive-deps

  // -- Node drawing (three.js) --
  // Return a fresh mesh per node. The library (three-forcegraph) owns the
  // object lifecycle — it disposes geometry/material when nodes are removed.
  // We access meshes for opacity sync via the library's `__threeObj` binding.

  const nodeThreeObject = useCallback((node: any) => {
    const n = node as GNode;
    const r = nodeRadius(n);
    const color = nodeColor(n);
    // People and projects render as octahedra so they stand out from the
    // systems and concepts that make up most of the graph. Both shapes come
    // from a unit geometry shared across every node and scaled to size.
    const faceted = n.type === "person" || n.type === "project";
    const material = new THREE.MeshPhongMaterial({
      color,
      specular: 0x222222,
      shininess: 25,
      transparent: true,
      opacity: 1,
    });
    const mesh = new THREE.Mesh(faceted ? SHARED_OCTAHEDRON : SHARED_SPHERE, material);
    mesh.scale.setScalar(faceted ? r * 1.6 : r);

    // No label sprite here — ensureLabel() builds one on the frame it first
    // needs to be visible. See the comment on ensureLabel.
    const group = new THREE.Group();
    group.add(mesh);
    (group as any).__nodeMesh = mesh;
    (group as any).__labelText = n.label;
    (group as any).__labelColor = color;
    (group as any).__labelOffset = r + 2.2;
    return group;
  }, []);

  // Sync mesh opacity with highlight/dim state by reaching into each node's
  // library-assigned __threeObj group. Avoids retaining our own mesh refs
  // (which the library may have already disposed).
  useEffect(() => {
    for (const n of rawGraphData.nodes as any[]) {
      const group = n.__threeObj as THREE.Object3D | undefined;
      const mesh = (group as any)?.__nodeMesh as THREE.Mesh | undefined;
      if (!mesh) continue;
      const isLit = highlightNodes.has(n.id);
      const opacity = hasDimming && !isLit ? 0.08 : 1;
      const mat = mesh.material as THREE.Material | undefined;
      if (mat && mat.opacity !== opacity) mat.opacity = opacity;
    }
  }, [highlightNodes, hasDimming, rawGraphData]);

  // Labels: visible when a node is highlighted (selection fan-out) OR when the
  // camera is within PROXIMITY_THRESHOLD of the node. Driven by a rAF loop
  // because pan/rotate don't emit React events.
  const highlightNodesRef = useRef<Set<string>>(highlightNodes);
  useEffect(() => { highlightNodesRef.current = highlightNodes; }, [highlightNodes]);

  useEffect(() => {
    // Iterate the RENDERED nodes. This walked every loaded node on every
    // animation frame — with isolated entities included that was 23,565
    // getWorldPosition calls per frame for the ~2,500 actually drawn.
    if (!graphData.nodes.length || offscreen) return;
    const PROXIMITY_THRESHOLD = 90;
    const tmp = new THREE.Vector3();
    let raf = 0;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const fg = fgRef.current;
      if (!fg) return;
      const camera: THREE.Camera | undefined = fg.camera?.();
      if (!camera) return;
      const lit = highlightNodesRef.current;
      const anyLit = lit.size > 0;
      for (const n of graphData.nodes as any[]) {
        const group = n.__threeObj as THREE.Object3D | undefined;
        if (!group) continue;
        const sprite = (group as any).__nodeLabel as THREE.Sprite | undefined;

        let show: boolean;
        if (anyLit) {
          show = lit.has(n.id);
        } else {
          group.getWorldPosition(tmp);
          show = tmp.distanceTo(camera.position) < PROXIMITY_THRESHOLD;
        }

        // Build the sprite only when it first has to be shown; a node that
        // never gets close or highlighted never pays for one.
        if (show) ensureLabel(group).visible = true;
        else if (sprite) sprite.visible = false;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [graphData, offscreen]);

  // -- Link styling --

  const linkColorFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      const key = `${src}\x00${tgt}`;
      const highlighted = highlightLinks.has(key);
      const dimmed = hasDimming && !highlighted;
      return edgeColor(link.type, highlighted, dimmed);
    },
    [highlightLinks, hasDimming]
  );

  // Highlighted edges get multiple arrow cones evenly spaced along their
  // length, slowly flowing toward the (dominant) target. Direction and
  // magnitude are conveyed by the cone orientation + motion.
  const linkParticlesFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      return highlightLinks.has(`${src}\x00${tgt}`) ? 5 : 0;
    },
    [highlightLinks]
  );

  // One arrow cone per link — the library clones it for each particle and
  // orients them via lookAt() toward the direction of travel (three-forcegraph
  // applies lookAt for non-sphere geometries at updatePhotons).
  const particleArrowFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      if (!highlightLinks.has(`${src}\x00${tgt}`)) return undefined;
      const geo = new THREE.ConeGeometry(0.7, 2.0, 6);
      // Tip points along +Y by default; rotate so it points along +Z so the
      // tip faces the particle's direction of travel (source → target).
      geo.rotateX(Math.PI / 2);
      const { color, alpha } = splitRgba(edgeColor(link.type, true, false));
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: alpha,
      });
      return new THREE.Mesh(geo, mat);
    },
    [highlightLinks]
  );

  const nodeTooltip = useCallback((node: any) => {
    const n = node as GNode;
    const label = escapeHtml(n.label);
    const defLine = n.definition
      ? `<br/><span style="color:#cbd5e1;font-size:11px;display:inline-block;max-width:260px;white-space:normal">${escapeHtml(n.definition)}</span>`
      : "";
    const kind = escapeHtml(n.type || "entity");
    const color = KIND_COLOR[n.type] || KIND_COLOR.entity;
    return `<b style="color:${color}">${label}</b> <span style="color:#94a3b8;font-size:10px">(${kind})</span>`
      + `<br/><span style="color:#94a3b8;font-size:10px">${n.factCount || 0} facts</span>${defLine}`;
  }, []);

  // -- Hover handler (debounced 150ms) --

  const handleNodeHover = useCallback((node: any) => {
    const n = node ? (node as GNode) : null;
    setHoverNode(n);
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = setTimeout(() => {
      if (onNodeHover) onNodeHover(n);
    }, 150);
  }, [onNodeHover]);

  // -- Click handler with double-click detection --

  const handleNodeClick = useCallback((node: any) => {
    const n = node as GNode;
    const now = Date.now();
    const last = lastClickRef.current;

    if (last && last.nodeId === n.id && now - last.ts < 300) {
      // Double click
      lastClickRef.current = null;
      if (onNodeDoubleClick) onNodeDoubleClick(n.id);
      return;
    }

    lastClickRef.current = { nodeId: n.id, ts: now };

    const deselecting = activeNode?.id === n.id;
    if (deselecting) {
      setActiveNode(null);
      if (onNodeClick) onNodeClick(null);
    } else {
      setActiveNode(n);
      if (onNodeClick) onNodeClick(n.id);
    }
  }, [activeNode, onNodeClick, onNodeDoubleClick]);

  const handleBackgroundClick = useCallback(() => {
    setActiveNode(null);
    if (onNodeClick) onNodeClick(null);
  }, [onNodeClick]);

  // -- Edge filter toggle --

  const toggleFilter = useCallback((type: keyof EdgeFilters) => {
    setEdgeFilters((prev) => ({ ...prev, [type]: !prev[type] }));
  }, []);

  // Pin every node at its resting position when the simulation settles,
  // so the layout doesn't drift on subsequent data changes (e.g. filter toggles).
  const handleEngineStop = useCallback(() => {
    for (const n of rawGraphData.nodes as any[]) {
      if (typeof n.x === "number") n.fx = n.x;
      if (typeof n.y === "number") n.fy = n.y;
      if (typeof n.z === "number") n.fz = n.z;
    }
    // Persist the settled layout so the next reload starts from it instead of
    // re-running ~1.4s of simulation. Skipped when we started from a cached
    // layout — nothing moved, so there is nothing new to write.
    if (!layoutPresettled && layoutSigRef.current && rawGraphData.nodes.length) {
      writeStoredLayout(layoutSigRef.current, rawGraphData.nodes);
      setLayoutPresettled(true);
    }
  }, [rawGraphData, layoutPresettled]);

  // Remap middle-click-drag to PAN (default is DOLLY/zoom, redundant with scroll).
  // TrackballControls.mouseButtons = { LEFT: ROTATE, MIDDLE: DOLLY, RIGHT: PAN }
  // Runs once the graph ref is populated.
  useEffect(() => {
    if (loading) return;
    const fg = fgRef.current;
    if (!fg) return;
    const controls = fg.controls?.();
    if (controls?.mouseButtons) {
      controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
    }
  }, [loading]);

  // -- Lighting --
  // Default lights flatten Phong highlights with too much ambient.
  // Use dimmer ambient + two directional lights for proper shading.
  const lights = useMemo(() => {
    const ambient = new THREE.AmbientLight(0x404060, 1.2);
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(150, 200, 100);
    const fill = new THREE.DirectionalLight(0x9fb8ff, 0.5);
    fill.position.set(-150, -50, -120);
    return [ambient, key, fill];
  }, []);

  // -- Render --

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-background rounded-lg border border-border/30">
      {!loading && !error && (
        <div className="absolute inset-0">
          <ForceGraph3D
            ref={fgRef}
            width={dimensions.w}
            height={dimensions.h}
            graphData={graphData}
            nodeId="id"
            nodeLabel={nodeTooltip}
            nodeThreeObject={nodeThreeObject}
            linkColor={linkColorFn}
            linkWidth={0}
            linkOpacity={1}
            linkCurvature={0.15}
            linkDirectionalParticles={linkParticlesFn}
            linkDirectionalParticleSpeed={0.001}
            linkDirectionalParticleThreeObject={particleArrowFn}
            backgroundColor="rgba(0,0,0,0)"
            onNodeHover={handleNodeHover}
            onNodeClick={handleNodeClick}
            onBackgroundClick={handleBackgroundClick}
            onEngineStop={handleEngineStop}
            // A restored layout is already settled and fully pinned, so both
            // phases are pure waste. Otherwise: warmupTicks is synchronous and
            // blocks the first paint, so keep it to the handful of ticks that
            // stop the graph from appearing as a ball at the origin (~120ms)
            // and let the visible cooldown do the rest of the settling.
            cooldownTicks={layoutPresettled ? 0 : 150}
            cooldownTime={8000}
            warmupTicks={layoutPresettled ? 0 : 10}
            d3AlphaDecay={0.04}
            d3VelocityDecay={0.4}
            enableNodeDrag={false}
            controlType="trackball"
            lights={lights}
          />
        </div>
      )}

      {/* Legend — top-left, merged section colors + edge types */}
      {!loading && !error && (
        <div className="absolute top-2 left-2 z-10 bg-card/80 backdrop-blur-sm px-2 py-1.5 rounded text-[10px] text-muted-foreground space-y-1">
          {/* Vault section node colors */}
          <div className="font-semibold text-[9px] uppercase tracking-wider text-muted-foreground mb-0.5">Nodes</div>
          <div className="flex flex-col gap-0.5">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{background: "hsl(160,35%,50%)"}} />
              memory
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{background: "hsl(270,35%,55%)"}} />
              projects
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{background: "hsl(45,45%,55%)"}} />
              knowledge
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{background: "hsl(340,35%,55%)"}} />
              agents
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-2 h-2 flex-shrink-0 rotate-45" style={{background: "#F59E0B"}} />
              entity
            </span>
          </div>
          {/* Edge category colors */}
          <div className="font-semibold text-[9px] uppercase tracking-wider text-muted-foreground mt-1 mb-0.5">Edges</div>
          <div className="flex flex-col gap-0.5">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: `rgba(${CATEGORY_COLOR.structural},0.8)`}} />
              structural
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: `rgba(${CATEGORY_COLOR.lineage},0.8)`}} />
              lineage
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: `rgba(${CATEGORY_COLOR.comparison},0.8)`}} />
              comparison
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: `rgba(${CATEGORY_COLOR.reference},0.8)`}} />
              reference
            </span>
          </div>
        </div>
      )}

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs">
          Loading knowledge graph...
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-400 text-xs">
          <p>{error}</p>
          <button onClick={() => loadGraph(true)} className="underline text-muted-foreground hover:text-foreground">Retry</button>
        </div>
      )}

      {/* Controls: filter toggles + reset/fit */}
      {!loading && !error && (
        <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1">
          {/* Node-kind legend. Keyed to app/entity_kind.py, so it names what
              the colours actually mean — the previous legend described edge
              categories only, while node colour came from a hash of the id. */}
          <div className="flex items-center gap-2 mr-2 bg-card/80 backdrop-blur-sm px-2 py-1 rounded text-[10px]">
            {KIND_ORDER.filter((k) => kindsPresent.has(k)).map((k) => (
              <span key={k} className="flex items-center gap-1 text-muted-foreground">
                <span className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                      style={{ background: KIND_COLOR[k] }} />
                {k}
              </span>
            ))}
          </div>

          {/* Edge type filter toggles */}
          <div className="flex items-center gap-2 mr-2 bg-card/80 backdrop-blur-sm px-2 py-1 rounded text-[10px]">
            {(["structural", "lineage", "comparison", "reference"] as const).map((cat) => (
              <label key={cat} className="flex items-center gap-1 cursor-pointer select-none text-muted-foreground hover:text-foreground">
                <input
                  type="checkbox"
                  checked={edgeFilters[cat]}
                  onChange={() => toggleFilter(cat)}
                  className="w-2.5 h-2.5 accent-primary cursor-pointer"
                />
                <span>{cat}</span>
              </label>
            ))}
          </div>
          <button
            onClick={() => {
              setActiveNode(null);
              if (onNodeClick) onNodeClick(null);
            }}
            title="Reset selection"
            className="p-1 rounded text-muted-foreground hover:text-foreground/90 hover:bg-secondary/80 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
          <button
            onClick={() => { if (fgRef.current) fgRef.current.zoomToFit(1200, 50); }}
            title="Fit all"
            className="p-1 rounded text-muted-foreground hover:text-foreground/90 hover:bg-secondary/80 transition-colors"
          >
            <Maximize className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
}
