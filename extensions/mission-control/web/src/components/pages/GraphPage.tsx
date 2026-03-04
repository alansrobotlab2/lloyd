import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { forceCollide, forceRadial, forceX, forceY } from "d3-force";
import { api, type TagGraphNode, type TagGraphEdge } from "../../api";
import { Search, RotateCcw, Maximize } from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────────

interface GNode {
  id: string;
  label: string;
  count: number;
  isHub: boolean;
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

// ── Brain emoji for the hub ─────────────────────────────────────────────

const BRAIN = "\u{1F9E0}";
const HUB_ID = "__hub__";

// ── Hue-based coloring: deterministic color per tag ─────────────────────

function tagColor(tag: string): string {
  let hash = 0;
  for (let i = 0; i < tag.length; i++) hash = (hash * 31 + tag.charCodeAt(i)) | 0;
  const hue = ((hash % 360) + 360) % 360;
  return `hsl(${hue}, 30%, 55%)`;
}

// ── Node radius from count ──────────────────────────────────────────────

function tagRadius(count: number): number {
  // min 1.5, max 6
  return Math.max(1.5, Math.min(6, 1.5 + Math.sqrt(count) * 0.75));
}

// ── Component ───────────────────────────────────────────────────────────

export default function GraphPage() {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const initialFitDone = useRef(false);
  const minZoomRef = useRef(0.01);
  const animFrameRef = useRef<number>(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [graphData, setGraphData] = useState<GData>({ nodes: [], links: [] });
  const [search, setSearch] = useState("");
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
    // Hub node
    const hub: GNode = { id: HUB_ID, label: "Vault", count: 0, isHub: true };

    // Tag nodes
    const tagNodes: GNode[] = rawNodes.map((n) => ({
      id: n.id,
      label: `#${n.label}`,
      count: n.count,
      isHub: false,
    }));

    // Hub → every tag link
    const hubLinks: GLink[] = tagNodes.map((n) => ({
      source: HUB_ID,
      target: n.id,
      weight: 1,
    }));

    // Tag co-occurrence links
    const tagLinks: GLink[] = rawEdges.map((e) => ({
      source: e.source,
      target: e.target,
      weight: e.weight,
    }));

    return {
      nodes: [hub, ...tagNodes],
      links: [...hubLinks, ...tagLinks],
    };
  }

  // ── Highlight sets ────────────────────────────────────────────────────

  const { highlightNodes, highlightLinks } = useMemo(() => {
    const hn = new Set<string>();
    const hl = new Set<string>();

    // Highlight from selection (exclude hub — it fades with the background)
    if (selectedNode) {
      hn.add(selectedNode.id);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (src === HUB_ID || tgt === HUB_ID) continue;
        if (src === selectedNode.id) { hn.add(tgt); hl.add(`${src}→${tgt}`); }
        if (tgt === selectedNode.id) { hn.add(src); hl.add(`${src}→${tgt}`); }
      }
    }

    // Highlight from hover (when nothing is selected)
    if (hoverNode && !selectedNode) {
      hn.add(hoverNode.id);
      for (const l of graphData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (src === hoverNode.id) { hn.add(tgt); hl.add(`${src}→${tgt}`); }
        if (tgt === hoverNode.id) { hn.add(src); hl.add(`${src}→${tgt}`); }
      }
    }

    if (search) {
      const lower = search.toLowerCase();
      for (const n of graphData.nodes) {
        if (n.label.toLowerCase().includes(lower)) hn.add(n.id);
      }
    }

    return { highlightNodes: hn, highlightLinks: hl };
  }, [selectedNode, hoverNode, search, graphData]);

  const hasDimming = !!(selectedNode || hoverNode || search);

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

  // ── Initial load ──────────────────────────────────────────────────────

  useEffect(() => { loadGraph(); }, [loadGraph]);

  // ── Fit to view once after first data load ────────────────────────────

  useEffect(() => {
    if (!graphData.nodes.length || initialFitDone.current) return;
    // Wait for warmup ticks to settle, then fit
    const t = setTimeout(() => {
      if (fgRef.current) {
        fgRef.current.zoomToFit(1000, 50);
        initialFitDone.current = true;
        // Capture min zoom after animation completes
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

  // ── Force configuration helper ──────────────────────────────────────

  // activeIds: set of node IDs participating in the selection cluster
  // selectedId: the node pinned at center (excluded from radial force)
  const configureForces = useCallback((activeIds?: Set<string>, selectedId?: string) => {
    const fg = fgRef.current;
    if (!fg) return;

    const isActive = (node: any): boolean => {
      if (!activeIds) return true;
      const n = node as GNode;
      // During selection, hub is NOT active — subgraph is fully isolated
      return activeIds.has(n.id);
    };

    // ── Collision: prevent overlap ──────────────────────────────────
    fg.d3Force("collide",
      forceCollide()
        .radius((node: any) => {
          if (!isActive(node)) return 0;
          const n = node as GNode;
          return n.isHub ? 20 : tagRadius(n.count) + (activeIds ? 10 : 4);
        })
        .strength(0.5)
        .iterations(1)
    );

    // ── Charge: inverse-square repulsion between all (active) nodes ─
    fg.d3Force("charge")?.strength((node: any) => {
      if (!isActive(node)) return 0;
      // Stronger repulsion during selection so neighbors spread apart
      return activeIds ? -200 : -80;
    }).distanceMax(activeIds ? 200 : 400);

    // ── Links: spring attraction along edges ────────────────────────
    fg.d3Force("link")
      ?.distance((link: any) => {
        const src = typeof link.source === "string" ? link.source : link.source.id;
        const tgt = typeof link.target === "string" ? link.target : link.target.id;
        if (src === HUB_ID || tgt === HUB_ID) return activeIds ? 60 : 180;
        const w = link.weight || 1;
        return Math.max(20, 60 - w * 8);
      })
      .strength((link: any) => {
        const src = typeof link.source === "string" ? link.source : link.source.id;
        const tgt = typeof link.target === "string" ? link.target : link.target.id;
        if (activeIds) {
          const srcActive = src === HUB_ID || activeIds.has(src);
          const tgtActive = tgt === HUB_ID || activeIds.has(tgt);
          if (!srcActive || !tgtActive) return 0;
        }
        if (src === HUB_ID || tgt === HUB_ID) return activeIds ? 0 : 0.015;
        const w = link.weight || 1;
        return Math.min(0.6, 0.15 + w * 0.08);
      });

    // ── Center gravity: weak pull toward origin ─────────────────────
    if (activeIds) {
      // During selection, pull active neighbors toward center
      fg.d3Force("centerX", forceX(0).strength((node: any) => {
        if (!isActive(node) || (node as GNode).isHub) return 0;
        return node.id === selectedId ? 0 : 0.05;
      }));
      fg.d3Force("centerY", forceY(0).strength((node: any) => {
        if (!isActive(node) || (node as GNode).isHub) return 0;
        return node.id === selectedId ? 0 : 0.05;
      }));
    } else {
      fg.d3Force("centerX", null);
      fg.d3Force("centerY", null);
    }

    // ── Radial force: distribute neighbors evenly around selected ────
    if (activeIds && selectedId) {
      const neighborCount = activeIds.size - 1; // exclude selected node
      const baseRadius = Math.max(40, 15 + neighborCount * 4);
      fg.d3Force("radial",
        forceRadial(
          (node: any) => {
            const n = node as GNode;
            // Larger nodes (higher count) orbit closer, smaller ones farther
            const r = tagRadius(n.count);
            const factor = 1.4 - 0.8 * (r - 1.5) / (6 - 1.5);
            return baseRadius * factor;
          },
          0, 0
        )
          .strength((node: any) => {
            const n = node as GNode;
            if (n.isHub || n.id === selectedId) return 0;
            return activeIds.has(n.id) ? 0.8 : 0;
          })
      );
      // Disable built-in center force — it pulls ALL nodes including background
      fg.d3Force("center", null);
    } else {
      fg.d3Force("radial", null);
      // Re-enable built-in center force for normal view
      // (react-force-graph-2d recreates it on graphData change, but be safe)
    }
  }, []);

  // ── Pin hub to center + initial force setup ────────────────────────────

  useEffect(() => {
    if (!graphData.nodes.length) return;
    const hub = graphData.nodes.find((n) => n.id === HUB_ID);
    if (hub) {
      (hub as any).fx = 0;
      (hub as any).fy = 0;
    }
    configureForces();
    fgRef.current?.d3ReheatSimulation();
  }, [graphData, configureForces]);

  // ── Node drawing ──────────────────────────────────────────────────────

  const nodeCanvasObject = useCallback(
    (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as GNode;
      const isLit = highlightNodes.has(n.id);
      const alpha = hasDimming && !isLit ? 0.12 : 1;

      if (n.isHub) {
        // Draw brain emoji
        const size = 28 / globalScale;
        ctx.globalAlpha = alpha;
        ctx.font = `${size}px serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(BRAIN, node.x, node.y);
        ctx.globalAlpha = 1;
        return;
      }

      // Tag node
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

      // Label: always show for large tags, at zoom for small ones
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
      // Generous hit area so small nodes are easy to hover/click
      const r = n.isHub ? 16 : Math.max(6, tagRadius(n.count) + 4);
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
      // Hub links are subtle
      if (src === HUB_ID || tgt === HUB_ID) {
        return hasDimming && !highlightLinks.has(`${src}→${tgt}`)
          ? "rgba(71,85,105,0.03)"
          : "rgba(100,116,139,0.2)";
      }
      if (highlightLinks.has(`${src}→${tgt}`)) return "rgba(168,183,204,0.85)";
      return hasDimming ? "rgba(71,85,105,0.04)" : "rgba(120,140,165,0.3)";
    },
    [highlightLinks, hasDimming]
  );

  const linkWidthFn = useCallback(
    (link: any) => {
      const src = typeof link.source === "string" ? link.source : link.source.id;
      const tgt = typeof link.target === "string" ? link.target : link.target.id;
      if (src === HUB_ID || tgt === HUB_ID) return 0.3;
      if (highlightLinks.has(`${src}→${tgt}`)) return 1.5;
      return Math.min(2, 0.3 + (link.weight || 1) * 0.2);
    },
    [highlightLinks]
  );

  // ── Tooltip ───────────────────────────────────────────────────────────

  const nodeLabel = useCallback((node: any) => {
    const n = node as GNode;
    if (n.isHub) return "<b>Vault Hub</b>";
    return `<b>${n.label}</b><br/><span style="color:#94a3b8;font-size:11px">${n.count} docs</span>`;
  }, []);

  // ── Render ────────────────────────────────────────────────────────────

  const tagCount = graphData.nodes.filter((n) => !n.isHub).length;
  const edgeCount = graphData.links.filter((l) => {
    const src = typeof l.source === "string" ? l.source : (l.source as any).id;
    return src !== HUB_ID;
  }).length;

  return (
    <div ref={containerRef} className="flex-1 relative min-h-0 overflow-hidden bg-surface-0">
      {/* Graph — always mounted so ResizeObserver works */}
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
              if (!node) { setHoverNode(null); return; }
              const n = node as GNode;
              setHoverNode(n.isHub ? null : n);
            }}
            onNodeClick={(node: any) => {
              const n = node as GNode;

              // Helper: release all non-hub nodes and restore full forces
              const releaseAll = () => {
                for (const nd of graphData.nodes) {
                  if (!nd.isHub) { (nd as any).fx = undefined; (nd as any).fy = undefined; }
                }
                configureForces();
              };

              if (n.isHub) {
                cancelAnimationFrame(animFrameRef.current);
                setSelectedNode(null);
                releaseAll();
                if (fgRef.current) {
                  fgRef.current.d3ReheatSimulation();
                  fgRef.current.zoomToFit(1200, 50);
                }
                return;
              }

              const deselecting = selectedNode?.id === n.id;
              setSelectedNode(deselecting ? null : n);

              if (fgRef.current) {
                cancelAnimationFrame(animFrameRef.current);
                if (deselecting) {
                  releaseAll();
                  fgRef.current.d3ReheatSimulation();
                  fgRef.current.zoomToFit(1200, 50);
                } else {
                  // Build neighbor set (excluding hub)
                  const neighborIds = new Set<string>([n.id]);
                  for (const l of graphData.links) {
                    const src = typeof l.source === "string" ? l.source : (l.source as any).id;
                    const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
                    if (src === n.id && tgt !== HUB_ID) neighborIds.add(tgt);
                    if (tgt === n.id && src !== HUB_ID) neighborIds.add(src);
                  }

                  // Compute circle targets for neighbors
                  const neighborArr = [...neighborIds].filter(id => id !== n.id);
                  const neighborCount = neighborArr.length;
                  const baseRadius = Math.max(40, 15 + neighborCount * 4);
                  const angleStep = (2 * Math.PI) / Math.max(1, neighborCount);

                  const nodeMap = new Map<string, GNode>();
                  for (const nd of graphData.nodes) nodeMap.set(nd.id, nd);
                  neighborArr.sort((a, b) => (nodeMap.get(b)?.count || 0) - (nodeMap.get(a)?.count || 0));

                  // Capture start positions for selected node
                  const fromX = node.x || 0;
                  const fromY = node.y || 0;
                  node.vx = 0;
                  node.vy = 0;

                  // Capture start positions and compute targets for neighbors
                  const neighborAnim: Array<{ nd: GNode; sx: number; sy: number; tx: number; ty: number }> = [];
                  for (const nd of graphData.nodes) {
                    if (nd.isHub || nd.id === n.id) continue;
                    if (neighborIds.has(nd.id)) {
                      const r = tagRadius(nd.count);
                      const factor = 1.4 - 0.8 * (r - 1.5) / (6 - 1.5);
                      const dist = baseRadius * factor;
                      const angle = angleStep * neighborArr.indexOf(nd.id);
                      neighborAnim.push({
                        nd,
                        sx: (nd as any).x || 0,
                        sy: (nd as any).y || 0,
                        tx: Math.cos(angle) * dist,
                        ty: Math.sin(angle) * dist,
                      });
                    }
                    // Freeze everything via fx/fy (neighbors will be moved by animation)
                    (nd as any).fx = (nd as any).x;
                    (nd as any).fy = (nd as any).y;
                    (nd as any).vx = 0;
                    (nd as any).vy = 0;
                  }

                  const fg = fgRef.current;

                  // Single animation: selected → center AND neighbors → circle, simultaneously
                  const animStart = performance.now();
                  const animDur = 1200;
                  const animTick = () => {
                    const t = Math.min(1, (performance.now() - animStart) / animDur);
                    const ease = 1 - (1 - t) * (1 - t); // ease-out quadratic

                    // Move selected node toward origin
                    node.fx = fromX * (1 - ease);
                    node.fy = fromY * (1 - ease);

                    // Move neighbors toward their circle targets
                    for (const { nd, sx, sy, tx, ty } of neighborAnim) {
                      (nd as any).fx = sx + (tx - sx) * ease;
                      (nd as any).fy = sy + (ty - sy) * ease;
                    }

                    if (t < 1) {
                      animFrameRef.current = requestAnimationFrame(animTick);
                    } else {
                      // Anchor selected node at center
                      node.fx = 0;
                      node.fy = 0;

                      // Unpin neighbors, let forces settle them gently
                      for (const { nd } of neighborAnim) {
                        (nd as any).vx = 0;
                        (nd as any).vy = 0;
                        (nd as any).fx = undefined;
                        (nd as any).fy = undefined;
                      }

                      configureForces(neighborIds, n.id);
                      if (fg) fg.d3ReheatSimulation();
                    }
                  };
                  animFrameRef.current = requestAnimationFrame(animTick);

                  // Zoom to fit the cluster during the animation
                  if (fg) fg.zoomToFit(1200, 80, (nd: any) => neighborIds.has(nd.id));
                }
              }
            }}
            onBackgroundClick={() => {
              cancelAnimationFrame(animFrameRef.current);
              setSelectedNode(null);
              for (const nd of graphData.nodes) {
                if (!nd.isHub) { (nd as any).fx = undefined; (nd as any).fy = undefined; }
              }
              configureForces();
              if (fgRef.current) {
                fgRef.current.d3ReheatSimulation();
                fgRef.current.zoomToFit(1200, 50);
              }
            }}
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

      {/* Loading / Error overlays */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center text-slate-400">
          Loading tag graph...
        </div>
      )}
      {error && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-red-400">
          <p>{error}</p>
          <button onClick={loadGraph} className="text-sm underline text-slate-400 hover:text-slate-200">
            Retry
          </button>
        </div>
      )}

      {/* Toolbar — overlaid on top */}
      {!loading && !error && (
        <div className="absolute top-0 left-0 right-0 z-10 flex items-center gap-3 px-4 py-2 bg-surface-0/80 backdrop-blur-sm border-b border-surface-3/20">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search tags..."
              className="pl-7 pr-3 py-1.5 text-xs bg-surface-2 rounded-md border border-surface-3/50 text-slate-200 placeholder:text-slate-500 outline-none focus:border-brand-500/50 w-52"
            />
          </div>
          <button
            onClick={() => {
              setSelectedNode(null);
              setSearch("");
              if (fgRef.current) fgRef.current.zoomToFit(1200, 50);
            }}
            title="Reset view"
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-surface-2 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => { if (fgRef.current) fgRef.current.zoomToFit(1200, 50); }}
            title="Fit all nodes"
            className="p-1.5 rounded-md text-slate-400 hover:text-slate-200 hover:bg-surface-2 transition-colors"
          >
            <Maximize className="w-3.5 h-3.5" />
          </button>
          <span className="ml-auto text-[10px] text-slate-500">
            {tagCount} tags · {edgeCount} co-occurrences
          </span>
        </div>
      )}
    </div>
  );
}
