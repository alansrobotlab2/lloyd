import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import SpriteText from "three-spritetext";
import { forceCollide } from "d3-force";
import { api, type EntityGraphNode } from "../api";
import { RotateCcw, Maximize } from "lucide-react";
import { categoryOf, CATEGORY_COLOR } from "../lib/edgeCategories";

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

function nodeColor(node: GNode): string {
  if (node.type === "entity") return "#F59E0B"; // gold/amber for entity nodes
  const id = node.id;
  if (id.startsWith("facts/")) return "hsl(200, 50%, 55%)";
  if (id.startsWith("memory/")) return "hsl(160, 35%, 50%)";
  if (id.startsWith("projects/")) return "hsl(270, 35%, 55%)";
  if (id.startsWith("knowledge/")) return "hsl(45, 45%, 55%)";
  if (id.startsWith("agents/")) return "hsl(340, 35%, 55%)";
  // Hash-based fallback
  let hash = 0;
  for (let i = 0; i < id.length; i++) hash = (hash * 31 + id.charCodeAt(i)) | 0;
  const hue = ((hash % 360) + 360) % 360;
  return `hsl(${hue}, 30%, 50%)`;
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
  if (dimmed) return "rgba(71,85,105,0.05)";
  const rgb = CATEGORY_COLOR[categoryOf(type)];
  // Highlighted: strong alpha so the selected subgraph pops.
  // Default (no selection): very faint so highlighted edges stand out.
  const alpha = highlighted ? 0.9 : 0.06;
  return `rgba(${rgb},${alpha})`;
}

// -- Component --

export default function EntityGraph({ selectedNode: selectedNodeId, onNodeClick, onNodeDoubleClick, onNodeHover }: EntityGraphProps) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const initialFitDone = useRef(false);
  const animFrameRef = useRef<number>(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rawGraphData, setRawGraphData] = useState<GData>({ nodes: [], links: [] });
  const [activeNode, setActiveNode] = useState<GNode | null>(null);
  const [hoverNode, setHoverNode] = useState<GNode | null>(null);
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });
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

  const loadGraph = useCallback(async () => {
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
      setRawGraphData({
        nodes,
        links: data.edges as GLink[],
      });
    } catch (err: any) {
      setError(err.message || "Failed to load entity graph");
    } finally {
      setLoading(false);
    }
  }, []);

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

  // -- Resize tracking --

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      setDimensions({ w: rect.width, h: rect.height });
    });
    ro.observe(container);
    const rect = container.getBoundingClientRect();
    setDimensions({ w: rect.width, h: rect.height });
    return () => ro.disconnect();
  }, []);

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
      fg.d3Force("collide",
        forceCollide().radius(() => 8).strength(0.6).iterations(2)
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
    const geometry = n.type === "entity"
      ? new THREE.OctahedronGeometry(r * 1.6, 0)
      : new THREE.SphereGeometry(r, 20, 16);
    const material = new THREE.MeshPhongMaterial({
      color,
      specular: 0x222222,
      shininess: 25,
      transparent: true,
      opacity: 1,
    });
    const mesh = new THREE.Mesh(geometry, material);

    const sprite = new SpriteText(n.label);
    sprite.textHeight = 2.2;
    sprite.color = n.type === "entity" ? "#FCD34D" : "#e2e8f0";
    sprite.backgroundColor = "rgba(15,23,42,0.65)";
    sprite.padding = 1.5;
    sprite.borderRadius = 2;
    sprite.position.set(0, r + 2.2, 0);
    sprite.visible = false;
    // Don't intercept pointer events targeting the node behind the label.
    (sprite as any).raycast = () => {};

    const group = new THREE.Group();
    group.add(mesh);
    group.add(sprite);
    (group as any).__nodeMesh = mesh;
    (group as any).__nodeLabel = sprite;
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
    if (!rawGraphData.nodes.length) return;
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
      for (const n of rawGraphData.nodes as any[]) {
        const group = n.__threeObj as THREE.Object3D | undefined;
        const sprite = (group as any)?.__nodeLabel as THREE.Sprite | undefined;
        if (!sprite || !group) continue;
        if (anyLit) {
          sprite.visible = lit.has(n.id);
          continue;
        }
        group.getWorldPosition(tmp);
        sprite.visible = tmp.distanceTo(camera.position) < PROXIMITY_THRESHOLD;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [rawGraphData]);

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
    if (n.type === "entity") {
      return `<b>⬥ ${label}</b> (entity)<br/><span style="color:#94a3b8;font-size:10px">${n.factCount || 0} facts</span>${defLine}`;
    }
    return `<b>${label}</b><br/><span style="color:#94a3b8;font-size:10px">${escapeHtml(n.id)}</span>${defLine}`;
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
  }, [rawGraphData]);

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
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-surface-0 rounded-lg border border-surface-3/30">
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
            cooldownTicks={100}
            cooldownTime={8000}
            warmupTicks={60}
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
        <div className="absolute top-2 left-2 z-10 bg-surface-1/80 backdrop-blur-sm px-2 py-1.5 rounded text-[10px] text-slate-400 space-y-1">
          {/* Vault section node colors */}
          <div className="font-semibold text-[9px] uppercase tracking-wider text-slate-500 mb-0.5">Nodes</div>
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
          <div className="font-semibold text-[9px] uppercase tracking-wider text-slate-500 mt-1 mb-0.5">Edges</div>
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
        <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
          Loading knowledge graph...
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-400 text-xs">
          <p>{error}</p>
          <button onClick={loadGraph} className="underline text-slate-400 hover:text-slate-200">Retry</button>
        </div>
      )}

      {/* Controls: filter toggles + reset/fit */}
      {!loading && !error && (
        <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1">
          {/* Edge type filter toggles */}
          <div className="flex items-center gap-2 mr-2 bg-surface-1/80 backdrop-blur-sm px-2 py-1 rounded text-[10px]">
            {(["structural", "lineage", "comparison", "reference"] as const).map((cat) => (
              <label key={cat} className="flex items-center gap-1 cursor-pointer select-none text-slate-400 hover:text-slate-200">
                <input
                  type="checkbox"
                  checked={edgeFilters[cat]}
                  onChange={() => toggleFilter(cat)}
                  className="w-2.5 h-2.5 accent-brand-500 cursor-pointer"
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
            className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-surface-2/80 transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
          </button>
          <button
            onClick={() => { if (fgRef.current) fgRef.current.zoomToFit(1200, 50); }}
            title="Fit all"
            className="p-1 rounded text-slate-500 hover:text-slate-300 hover:bg-surface-2/80 transition-colors"
          >
            <Maximize className="w-3 h-3" />
          </button>
        </div>
      )}
    </div>
  );
}
