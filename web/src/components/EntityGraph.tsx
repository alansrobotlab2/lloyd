import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ForceGraph3D from "react-force-graph-3d";
import * as THREE from "three";
import { forceCollide } from "d3-force";
import { api, type EntityGraphNode } from "../api";
import { RotateCcw, Maximize } from "lucide-react";

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

// Edge filter state
interface EdgeFilters {
  "wiki-link": boolean;
  "tag-cluster": boolean;
  "has-facts": boolean;
}

// -- Helpers --

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
  return Math.max(2, Math.min(8, 2 + Math.sqrt(degree) * 0.6));
}

function edgeColor(type: string, highlighted: boolean, dimmed: boolean): string {
  if (dimmed) return "rgba(71,85,105,0.06)";
  if (highlighted) {
    if (type === "tag-cluster") return "rgba(251,191,36,0.9)";
    if (type === "has-facts") return "rgba(245,158,11,0.9)";
    return "rgba(148,163,184,0.9)";
  }
  if (type === "tag-cluster") return "rgba(251,191,36,0.35)";
  if (type === "has-facts") return "rgba(245,158,11,0.40)";
  return "rgba(100,116,139,0.40)";
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
    "wiki-link": true,
    "tag-cluster": true,
    "has-facts": true,
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
      const type = l.type as keyof EdgeFilters;
      return edgeFilters[type] !== false;
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
        forceCollide().radius(() => 4).strength(0.4).iterations(1)
      );
      fg.d3Force("charge")?.strength(-40).distanceMax(300);
      fg.d3Force("link")?.distance(() => 25).strength(0.3);
    } catch (e) {
      console.warn("configureForces failed:", e);
    }
  }, []);

  useEffect(() => {
    if (!rawGraphData.nodes.length) return;
    // Defer to next tick so ForceGraph3D's internal simulation is initialized
    const t = setTimeout(() => {
      configureForces();
      fgRef.current?.d3ReheatSimulation();
    }, 0);
    return () => clearTimeout(t);
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
    return new THREE.Mesh(geometry, material);
  }, []);

  // Sync mesh opacity with highlight/dim state by reaching into each node's
  // library-assigned __threeObj. Avoids retaining our own mesh references
  // (which the library may have already disposed).
  useEffect(() => {
    for (const n of rawGraphData.nodes as any[]) {
      const mesh = n.__threeObj as THREE.Mesh | undefined;
      if (!mesh) continue;
      const isLit = highlightNodes.has(n.id);
      const opacity = hasDimming && !isLit ? 0.08 : 1;
      const mat = mesh.material as THREE.Material | undefined;
      if (mat && mat.opacity !== opacity) mat.opacity = opacity;
    }
  }, [highlightNodes, hasDimming, rawGraphData]);

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

  const nodeTooltip = useCallback((node: any) => {
    const n = node as GNode;
    if (n.type === "entity") {
      return `<b>⬥ ${n.label}</b> (entity)<br/><span style="color:#94a3b8;font-size:10px">${n.factCount || 0} facts</span>`;
    }
    return `<b>${n.label}</b><br/><span style="color:#94a3b8;font-size:10px">${n.id}</span>`;
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
            linkOpacity={0.5}
            linkCurvature={0.15}
            backgroundColor="rgba(0,0,0,0)"
            onNodeHover={handleNodeHover}
            onNodeClick={handleNodeClick}
            onBackgroundClick={handleBackgroundClick}
            cooldownTicks={100}
            cooldownTime={8000}
            d3AlphaDecay={0.04}
            d3VelocityDecay={0.4}
            enableNodeDrag={true}
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
          {/* Edge type colors */}
          <div className="font-semibold text-[9px] uppercase tracking-wider text-slate-500 mt-1 mb-0.5">Edges</div>
          <div className="flex flex-col gap-0.5">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: "rgba(148,163,184,0.7)"}} />
              wiki-link
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: "rgba(251,191,36,0.7)"}} />
              tag-cluster
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 h-px flex-shrink-0" style={{background: "rgba(245,158,11,0.7)"}} />
              has-facts
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
            {(["wiki-link", "tag-cluster", "has-facts"] as const).map((type) => (
              <label key={type} className="flex items-center gap-1 cursor-pointer select-none text-slate-400 hover:text-slate-200">
                <input
                  type="checkbox"
                  checked={edgeFilters[type]}
                  onChange={() => toggleFilter(type)}
                  className="w-2.5 h-2.5 accent-brand-500 cursor-pointer"
                />
                <span>{type}</span>
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
