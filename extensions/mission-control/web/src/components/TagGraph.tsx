import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { forceCollide, forceRadial, forceX, forceY } from "d3-force";
import { api, type TagGraphNode, type TagGraphEdge } from "../api";
import { RotateCcw, Maximize } from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────────

interface GNode {
  id: string;
  label: string;
  count: number;
}

interface GLink {
  source: string;
  target: string;
  weight: number;
}

interface GData {
  nodes: GNode[];
  links: GLink[];
}

interface TagGraphProps {
  /** External selection — drives the graph to select this tag node */
  selectedTag?: string | null;
  /** Called when a tag node is clicked or deselected (null = deselected) */
  onTagClick?: (tag: string | null) => void;
}

// ── Helpers ─────────────────────────────────────────────────────────────

function tagColor(tag: string): string {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) hash = (hash * 31 + tag.charCodeAt(i)) | 0;
  const hue = ((hash % 360) + 360) % 360;
  return `hsl(${hue}, 30%, 55%)`;
}

function tagRadius(count: number): number {
  return Math.max(1.5, Math.min(6, 1.5 + Math.sqrt(count) * 0.75));
}

// ── Component ───────────────────────────────────────────────────────────

export default function TagGraph({ selectedTag, onTagClick }: TagGraphProps) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const initialFitDone = useRef(false);
  const minZoomRef = useRef(0.01);
  const animFrameRef = useRef<number>(0);
  // Tracks the last selection we drove (from click or prop), to prevent loops
  const lastSelectedRef = useRef<string | null>(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GData>({ nodes: [], links: [] });
  const [selectedNode, setSelectedNode] = useState<GNode | null>(null);
  const [hoverNode, setHoverNode] = useState<GNode | null>(null);
  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });
  const [minZoom, setMinZoom] = useState(0.01);

  // ── Load data ───────────────────────────────────────────────────────

  const loadGraph = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.tagGraph();
      const gd = buildGraphData(data.nodes, data.edges);
      setGraphData(gd);
    } catch (err: any) {
      setError(err.message || "Failed to load graph");
    } finally {
      setLoading(false);
    }
  }, []);

  function buildGraphData(rawNodes: TagGraphNode[], rawEdges: TagGraphEdge[]): GData {
    const tagNodes: GNode[] = rawNodes.map((n) => ({
      id: n.id,
      label: `#${n.label}`,
      count: n.count,
    }));
    const tagLinks: GLink[] = rawEdges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
    }));
    return { nodes: tagNodes, links: tagLinks };
  }

  // ── Highlight sets ────────────────────────────────────────────────────

  const { highlightNodes, highlightLinks } = useMemo(() => {
    const hn = new Set<string>();
    const hl = new Set<string>();
    if (selectedNode) {
      hn.add(selectedNode.id);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (src === selectedNode.id) { hn.add(tgt); hl.add(`${src}→${tgt}`); }
        if (tgt === selectedNode.id) { hn.add(src); hl.add(`${src}→${tgt}`); }
      }
    }
    if (hoverNode && !selectedNode) {
      hn.add(hoverNode.id);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (src === hoverNode.id) { hn.add(tgt); hl.add(`${src}→${tgt}`); }
        if (tgt === hoverNode.id) { hn.add(src); hl.add(`${src}→${tgt}`); }
      }
    }
    return { highlightNodes: hn, highlightLinks: hl };
  }, [selectedNode, hoverNode, graphData]);

  const hasDimming = !!(selectedNode || hoverNode);

  // ── Resize tracking ───────────────────────────────────────────────────

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

  useEffect(() => { loadGraph(); }, [loadGraph]);

  // ── Fit to view once after first data load ────────────────────────────

  useEffect(() => {
    if (!graphData.nodes.length || initialFitDone.current) return;
    const t = setTimeout(() => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(1000, 50);
        initialFitDone.current = true;
        setTimeout(() => {
          if (fgRef.current) {
            const fitZoom = fgRef.current.zoom();
            minZoomRef.current = fitZoom;
            setMinZoom(fitZoom * 0.95);
          }
        }, 1100);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [graphData]);

  // ── Force configuration ──────────────────────────────────────────────

  const configureForces = useCallback((activeIds?: Set<string>, selectedId?: string) => {
    const fg = fgRef.current;
    if (!fg) return;

    const isActive = (node: any): boolean => {
      if (!activeIds) return true;
      return activeIds.has((node as GNode).id);
    };

    fg.d3Force("collide",
      forceCollide()
        .radius((node: any) => {
          if (!isActive(node)) return 0;
          return tagRadius((node as GNode).count) + (activeIds ? 10 : 4);
        })
        .strength(0.5)
        .iterations(1)
    );

    fg.d3Force("charge")?.strength((node: any) => {
      if (!isActive(node)) return 0;
      return activeIds ? -200 : -80;
    }).distanceMax(activeIds ? 200 : 400);

    fg.d3Force("link")
      ?.distance((link: any) => {
        const w = link.weight || 1;
        return Math.max(20, 60 - w * 8);
      })
      .strength((link: any) => {
        if (activeIds) {
          const src = typeof link.source === "string" ? link.source : link.source.id;
          const tgt = typeof link.target === "string" ? link.target : link.target.id;
          if (!activeIds.has(src) || !activeIds.has(tgt)) return 0;
        }
        const w = link.weight || 1;
        return Math.min(0.6, 0.15 + w * 0.08);
      });

    if (activeIds) {
      fg.d3Force("centerX", forceX(0).strength((node: any) => {
        if (!isActive(node)) return 0;
        return node.id === selectedId ? 0 : 0.05;
      }));
      fg.d3Force("centerY", forceY(0).strength((node: any) => {
        if (!isActive(node)) return 0;
        return node.id === selectedId ? 0 : 0.05;
      }));
    } else {
      fg.d3Force("centerX", null);
      fg.d3Force("centerY", null);
    }

    if (activeIds && selectedId) {
      const neighborCount = activeIds.size - 1;
      const baseRadius = Math.max(40, 15 + neighborCount * 4);
      fg.d3Force("radial",
        forceRadial(
          (node: any) => {
            const r = tagRadius((node as GNode).count);
            const factor = 1.4 - 0.8 * (r - 1.5) / (6 - 1.5);
            return baseRadius * factor;
          },
          0, 0
        )
          .strength((node: any) => {
            if (node.id === selectedId) return 0;
            return activeIds.has(node.id) ? 0.8 : 0;
          })
      );
      fg.d3Force("center", null);
    } else {
      fg.d3Force("radial", null);
    }
  }, []);

  // ── Initial forces ─────────────────────────────────────────────────────

  useEffect(() => {
    if (!graphData.nodes.length) return;
    configureForces();
    fgRef.current?.d3ReheatSimulation();
  }, [graphData, configureForces]);

  // ── Core animation helpers ────────────────────────────────────────────

  const animateDeselect = useCallback(() => {
    cancelAnimationFrame(animFrameRef.current);
    setSelectedNode(null);
    for (const nd of graphData.nodes) {
      (nd as any).fx = undefined; (nd as any).fy = undefined;
    }
    configureForces();
    if (fgRef.current) {
      fgRef.current.d3ReheatSimulation();
      fgRef.current.zoomToFit(1200, 50);
    }
  }, [graphData, configureForces]);

  const animateSelectById = useCallback((tagId: string) => {
    // Find the d3 node (graphData.nodes are mutated in place by d3)
    const n = graphData.nodes.find((nd) => nd.id === tagId);
    if (!n) return;

    cancelAnimationFrame(animFrameRef.current);
    setSelectedNode(n);

    if (!fgRef.current) return;

    const neighborIds = new Set<string>([n.id]);
    for (const l of graphData.links) {
      const src = typeof l.source === "string" ? l.source : (l.source as any).id;
      const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
      if (src === n.id) neighborIds.add(tgt);
      if (tgt === n.id) neighborIds.add(src);
    }

    const neighborArr = [...neighborIds].filter(id => id !== n.id);
    const neighborCount = neighborArr.length;
    const baseRadius = Math.max(40, 15 + neighborCount * 4);
    const angleStep = (2 * Math.PI) / Math.max(1, neighborCount);

    const nodeMap = new Map<string, GNode>();
    for (const nd of graphData.nodes) nodeMap.set(nd.id, nd);
    neighborArr.sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));

    // The d3 node has .x/.y from simulation
    const node = n as any;
    const fromX = node.x || 0;
    const fromY = node.y || 0;
    node.vx = 0;
    node.vy = 0;

    // Compute screen-center in graph coordinates so the node ends up visible
    const fg = fgRef.current;
    let centerX = 0, centerY = 0;
    if (fg) {
      const { x, y } = fg.screen2GraphCoords(dimensions.w / 2, dimensions.h / 2);
      centerX = x;
      centerY = y;
    }

    const neighborAnim: Array<{ nd: GNode; sx: number; sy: number; tx: number; ty: number }> = [];
    for (const nd of graphData.nodes) {
      if (nd.id === n.id) continue;
      if (neighborIds.has(nd.id)) {
        const r = tagRadius(nd.count);
        const factor = 1.4 - 0.8 * (r - 1.5) / (6 - 1.5);
        const dist = baseRadius * factor;
        const angle = angleStep * neighborArr.indexOf(nd.id);
        neighborAnim.push({
          nd,
          sx: (nd as any).x || 0,
          sy: (nd as any).y || 0,
          tx: centerX + Math.cos(angle) * dist,
          ty: centerY + Math.sin(angle) * dist,
        });
      }
      (nd as any).fx = (nd as any).x;
      (nd as any).fy = (nd as any).y;
      (nd as any).vx = 0;
      (nd as any).vy = 0;
    }

    const animStart = performance.now();
    const animDur = 1200;
    const animTick = () => {
      const t = Math.min(1, (performance.now() - animStart) / animDur);
      const ease = 1 - (1 - t) * (1 - t);
      node.fx = fromX + (centerX - fromX) * ease;
      node.fy = fromY + (centerY - fromY) * ease;
      for (const { nd, sx, sy, tx, ty } of neighborAnim) {
        (nd as any).fx = sx + (tx - sx) * ease;
        (nd as any).fy = sy + (ty - sy) * ease;
      }
      if (t < 1) {
        animFrameRef.current = requestAnimationFrame(animTick);
      } else {
        node.fx = centerX;
        node.fy = centerY;
        // Keep neighbors pinned at their arranged positions
        for (const { nd, tx, ty } of neighborAnim) {
          (nd as any).vx = 0;
          (nd as any).vy = 0;
          (nd as any).fx = tx;
          (nd as any).fy = ty;
        }
        configureForces(neighborIds, n.id);
        if (fg) fg.d3ReheatSimulation();
      }
    };
    animFrameRef.current = requestAnimationFrame(animTick);
    // Pan camera to center on the selected node's destination
    if (fg) fg.centerAt(centerX, centerY, 1200);
  }, [graphData, configureForces]);

  // ── Sync with external selectedTag prop ────────────────────────────────

  useEffect(() => {
    // Don't act until graph is loaded
    if (!graphData.nodes.length || loading) return;

    const incoming = selectedTag ?? null;
    // If we already drove this selection (from click or previous prop), skip
    if (incoming === lastSelectedRef.current) return;
    lastSelectedRef.current = incoming;

    if (incoming === null) {
      if (selectedNode) animateDeselect();
    } else {
      animateSelectById(incoming);
    }
  }, [selectedTag, graphData, loading, selectedNode, animateDeselect, animateSelectById]);

  // ── Node drawing ──────────────────────────────────────────────────────

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GNode;
      const isLit = highlightNodes.has(n.id);
      const alpha = hasDimming && !isLit ? 0.12 : 1;

      const r = tagRadius(n.count);
      const color = tagColor(n.id);
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.globalAlpha = alpha;
      ctx.fill();
      if (isLit && hasDimming) {
        ctx.strokeStyle = "#64748b";
        ctx.lineWidth = 2 / globalScale;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      const showLabel = globalScale > 1.2 || n.count >= 8 || (isLit && hasDimming);
      if (showLabel) {
        const fontSize = Math.max(10 / globalScale, 2);
        ctx.font = `${fontSize}px ui-monospace, monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.globalAlpha = hasDimming && !isLit ? 0.1 : 0.9;
        ctx.fillStyle = "#e2e8f0";
        ctx.fillText(n.label, node.x, node.y + r + 2 / globalScale);
        ctx.globalAlpha = 1;
      }
    },
    [highlightNodes, hasDimming]
  );

  const nodePointerAreaPaint = useCallback(
    (node: any, color: string, ctx: CanvasRenderingContext2D) => {
      const n = node as GNode;
      const r = Math.max(6, tagRadius(n.count) + 4);
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    },
    []
  );

  // ── Link styling ──────────────────────────────────────────────────────

  const linkColorFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      if (highlightLinks.has(`${src}→${tgt}`)) return "rgba(168,183,204,0.85)";
      return hasDimming ? "rgba(71,85,105,0.04)" : "rgba(120,140,165,0.3)";
    },
    [highlightLinks, hasDimming]
  );

  const linkWidthFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      if (highlightLinks.has(`${src}→${tgt}`)) return 1.5;
      return Math.min(2, 0.3 + (link.weight || 1) * 0.2);
    },
    [highlightLinks]
  );

  const nodeLabel = useCallback((node: any) => {
    const n = node as GNode;
    return `<b>${n.label}</b><br/><span style="color:#94a3b8;font-size:11px">${n.count} docs</span>`;
  }, []);

  // ── Click handlers ────────────────────────────────────────────────────

  const handleNodeClick = useCallback((node: any) => {
    const n = node as GNode;
    const deselecting = selectedNode?.id === n.id;
    const newTag = deselecting ? null : n.id;
    lastSelectedRef.current = newTag;

    if (deselecting) {
      animateDeselect();
    } else {
      animateSelectById(n.id);
    }

    if (onTagClick) onTagClick(newTag);
  }, [selectedNode, animateDeselect, animateSelectById, onTagClick]);

  const handleBackgroundClick = useCallback(() => {
    lastSelectedRef.current = null;
    animateDeselect();
    if (onTagClick) onTagClick(null);
  }, [animateDeselect, onTagClick]);

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div ref={containerRef} className="w-full h-full relative overflow-hidden bg-surface-0 rounded-lg border border-surface-3/30">
      {!loading && !error && (
        <div className="absolute inset-0">
          <ForceGraph2D
            ref={fgRef}
            width={dimensions.w}
            height={dimensions.h}
            graphData={graphData}
            nodeId="id"
            nodeLabel={nodeLabel}
            nodeCanvasObject={nodeCanvasObject}
            nodePointerAreaPaint={nodePointerAreaPaint}
            linkColor={linkColorFn}
            linkWidth={linkWidthFn}
            backgroundColor="transparent"
            onNodeHover={(node: any) => {
              setHoverNode(node ? (node as GNode) : null);
            }}
            onNodeClick={handleNodeClick}
            onBackgroundClick={handleBackgroundClick}
            minZoom={minZoom}
            cooldownTicks={400}
            d3AlphaDecay={0.025}
            d3VelocityDecay={0.8}
            enableNodeDrag={true}
            onNodeDragEnd={() => {
              if (fgRef.current) fgRef.current.d3ReheatSimulation();
            }}
            warmupTicks={80}
          />
        </div>
      )}

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-xs">
          Loading tag graph...
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-red-400 text-xs">
          <p>{error}</p>
          <button onClick={loadGraph} className="underline text-slate-400 hover:text-slate-200">Retry</button>
        </div>
      )}

      {/* Minimal controls — bottom-right corner */}
      {!loading && !error && (
        <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1">
          <button
            onClick={() => {
              lastSelectedRef.current = null;
              animateDeselect();
              if (onTagClick) onTagClick(null);
            }}
            title="Reset view"
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
