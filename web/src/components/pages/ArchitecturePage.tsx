import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import _ForceGraph3D from "react-force-graph-3d";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const ForceGraph3D = _ForceGraph3D as any;
import * as THREE from "three";
import SpriteText from "three-spritetext";
import { forceCollide } from "d3-force";
import {
  FileText,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  X,
  Code2,
  ArrowRightLeft,
  Layers,
  RotateCcw,
  Maximize,
} from "lucide-react";

// ── Types ───────────────────────────────────────────────────────────────

interface FileEntry {
  name: string;
  path: string;
  type: "file" | "dir";
  size?: number;
  children?: number;
}

interface TreeNode {
  name: string;
  path: string;
  type: "file" | "dir";
  size?: number;
  children?: number;
  entries?: TreeNode[];
  loaded?: boolean;
  expanded?: boolean;
}

interface FileContent {
  path: string;
  content: string;
  language: string;
  lineCount: number;
}

interface GraphNode {
  id: string;
  path: string;
  count: number;
  lang?: string;
  exports?: string[];
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
}

interface GraphLink {
  source: string;
  target: string;
  symbols?: string[];
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  totalImports: number;
  totalNodes: number;
  totalLinks: number;
}

// ── Utility Functions ───────────────────────────────────────────────────

// Dark-mode safe color tokens
const HC = {
  keyword: "#c084fc",
  literal: "#fb923c",
  number:  "#60a5fa",
  string:  "#4ade80",
  comment: "#64748b",
  type:    "#facc15",
};

function highlightCode(code: string, language: string): string {
  const escaped = code
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  const patterns: Record<string, Array<{ regex: RegExp; color: string }>> = {
    typescript: [
      { regex: /(\/\/.*$)/gm, color: HC.comment },
      { regex: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, color: HC.string },
      { regex: /\b(const|let|var|function|class|import|export|from|return|if|else|for|while|switch|case|break|continue|async|await|new|this|extends|implements|interface|type|namespace|enum)\b/g, color: HC.keyword },
      { regex: /\b(true|false|null|undefined)\b/g, color: HC.literal },
      { regex: /\b([A-Z][a-zA-Z0-9]*)\b/g, color: HC.type },
      { regex: /\b(\d+)\b/g, color: HC.number },
    ],
    javascript: [
      { regex: /(\/\/.*$)/gm, color: HC.comment },
      { regex: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, color: HC.string },
      { regex: /\b(const|let|var|function|class|import|export|from|return|if|else|for|while|switch|case|break|continue|async|await|new|this|extends|implements|interface|type|namespace|enum)\b/g, color: HC.keyword },
      { regex: /\b(true|false|null|undefined)\b/g, color: HC.literal },
      { regex: /\b(\d+)\b/g, color: HC.number },
    ],
    python: [
      { regex: /(#.*$)/gm, color: HC.comment },
      { regex: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, color: HC.string },
      { regex: /\b(def|class|import|from|return|if|elif|else|for|while|try|except|finally|with|as|async|await|lambda|yield|global|nonlocal)\b/g, color: HC.keyword },
      { regex: /\b(True|False|None)\b/g, color: HC.literal },
      { regex: /\b(\d+)\b/g, color: HC.number },
    ],
    markdown: [
      { regex: /^(#{1,6}\s.*)$/gm, color: HC.keyword },
      { regex: /(\*\*.*?\*\*)/g, color: HC.type },
      { regex: /(\[.*?\]\(.*?\))/g, color: HC.number },
    ],
    json: [
      { regex: /("(?:[^"\\]|\\.)*"\s*)(?=:)/g, color: HC.type },
      { regex: /(?<=:\s*)("(?:[^"\\]|\\.)*")/g, color: HC.string },
      { regex: /(?<=:\s*)(\d+)/g, color: HC.number },
      { regex: /(?<=:\s*)(true|false|null)\b/g, color: HC.literal },
    ],
  };

  const langPatterns = patterns[language] || patterns.python;

  // Single-pass: collect all non-overlapping matches sorted by position,
  // then build the output string with spans only around matched ranges.
  type Match = { start: number; end: number; color: string; text: string };
  const matches: Match[] = [];

  for (const { regex, color } of langPatterns) {
    let m: RegExpExecArray | null;
    regex.lastIndex = 0;
    while ((m = regex.exec(escaped)) !== null) {
      matches.push({ start: m.index, end: m.index + m[0].length, color, text: m[0] });
      if (m[0].length === 0) { regex.lastIndex++; }
    }
  }

  // Sort by start position; earlier wins, ties broken by longer match
  matches.sort((a, b) => a.start - b.start || b.end - a.end);

  // Build output, skipping overlapping matches
  let result = "";
  let pos = 0;
  for (const { start, end, color, text } of matches) {
    if (start < pos) continue; // overlaps a prior match — skip
    result += escaped.slice(pos, start);
    result += `<span style="color:${color}">${text}</span>`;
    pos = end;
  }
  result += escaped.slice(pos);
  return result;
}


function getFileName(path: string): string {
  return path.split("/").pop() || path;
}

function countFunctions(content: string, language: string): number {
  if (language === "python") {
    return (content.match(/^\s*def\s+/gm) || []).length;
  }
  const functionMatches = content.match(/\bfunction\s+/g) || [];
  const arrowMatches = content.match(/=>\s*\{/g) || [];
  return functionMatches.length + arrowMatches.length;
}

function countExports(content: string, language: string): number {
  if (language === "python") {
    return (content.match(/^__all__\s*=/m) ? 1 : 0) + (content.match(/^\s*class\s+/gm) || []).length;
  }
  return (content.match(/\bexport\s+/g) || []).length;
}

// ── File Tree Component ─────────────────────────────────────────────────

function FileTree({
  path,
  onOpenFile,
}: {
  path: string;
  onOpenFile: (path: string) => void;
}) {
  const [tree, setTree] = useState<TreeNode[]>([]);
  const [loading, setLoading] = useState(true);
  const treeRef = useRef<TreeNode[]>([]);
  treeRef.current = tree;

  useEffect(() => {
    setLoading(true);
    api.browse(path)
      .then((result) => {
        const nodes: TreeNode[] = result.entries.map((e: FileEntry) => ({
          ...e,
          loaded: false,
          expanded: false,
        }));
        setTree(nodes);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load file tree:", err);
        setLoading(false);
      });
  }, [path]);

  const toggleDir = useCallback(async (nodePath: string) => {
    const updateNodes = async (nodes: TreeNode[]): Promise<TreeNode[]> => {
      const result: TreeNode[] = [];
      for (const node of nodes) {
        if (node.path === nodePath && node.type === "dir") {
          if (!node.loaded) {
            try {
              const sub = await api.browse(nodePath);
              result.push({
                ...node,
                expanded: true,
                loaded: true,
                entries: sub.entries.map((e: FileEntry) => ({
                  ...e,
                  loaded: false,
                  expanded: false,
                })),
              });
            } catch {
              result.push(node);
            }
          } else {
            result.push({ ...node, expanded: !node.expanded });
          }
        } else if (node.entries) {
          result.push({ ...node, entries: await updateNodes(node.entries) });
        } else {
          result.push(node);
        }
      }
      return result;
    };
    const updated = await updateNodes(treeRef.current);
    setTree(updated);
  }, []);

  const renderNode = (node: TreeNode, depth: number) => {
    const indent = depth * 14;

    if (node.type === "dir") {
      return (
        <div key={node.path}>
          <button
            onClick={() => toggleDir(node.path)}
            className="w-full flex items-center gap-1.5 py-0.5 text-[11px] text-foreground/90 hover:bg-secondary rounded transition-colors"
            style={{ paddingLeft: `${indent + 4}px` }}
          >
            {node.expanded ? (
              <ChevronDown className="w-3 h-3 text-muted-foreground flex-shrink-0" />
            ) : (
              <ChevronRight className="w-3 h-3 text-muted-foreground flex-shrink-0" />
            )}
            {node.expanded ? (
              <FolderOpen className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
            ) : (
              <Folder className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
            )}
            <span className="truncate">{node.name}</span>
            {node.children != null && (
              <span className="text-[10px] text-muted-foreground/70 ml-auto pr-1">{node.children}</span>
            )}
          </button>
          {node.expanded && node.entries?.map((child) => renderNode(child, depth + 1))}
        </div>
      );
    }

    return (
      <button
        key={node.path}
        onClick={() => onOpenFile(node.path)}
        className="w-full flex items-center gap-1.5 py-0.5 text-[11px] text-muted-foreground hover:text-foreground hover:bg-secondary rounded transition-colors"
        style={{ paddingLeft: `${indent + 4 + 15}px` }}
      >
        <FileText className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
        <span className="truncate">{node.name}</span>
        {node.size != null && (
          <span className="text-[10px] text-muted-foreground/70 ml-auto pr-1">
            {node.size < 1024 ? `${node.size}B` : `${(node.size / 1024).toFixed(1)}K`}
          </span>
        )}
      </button>
    );
  };

  if (loading) {
    return <div className="text-[11px] text-muted-foreground px-2 py-4">Loading...</div>;
  }

  return <div className="space-y-0">{tree.map((node) => renderNode(node, 0))}</div>;
}

// ── File Viewer Modal ───────────────────────────────────────────────────

function FileViewerModal({
  file,
  onClose,
}: {
  file: FileContent;
  onClose: () => void;
}) {
  const lines = file.content.split("\n");

  return (
    <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-50 p-8">
      <div className="bg-card rounded-lg border border-border/50 overflow-hidden flex flex-col max-w-5xl w-full max-h-[90vh]">
        <div className="flex items-center justify-between px-4 py-2 border-b border-border/30 bg-secondary flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <Code2 className="w-4 h-4 text-primary flex-shrink-0" />
            <span className="text-sm font-medium text-foreground truncate">{file.path}</span>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-muted-foreground flex-shrink-0">
            <span>{file.language}</span>
            <span>{file.lineCount} lines</span>
            <button
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto font-mono text-xs">
          <div className="flex">
            <div className="w-12 flex-shrink-0 bg-secondary/50 text-muted-foreground/70 text-right pr-3 py-2 select-none border-r border-border/20">
              {lines.map((_, i) => (
                <div key={i} className="leading-5">{i + 1}</div>
              ))}
            </div>
            <div className="flex-1 p-2 overflow-x-auto">
              <pre className="leading-5 whitespace-pre-wrap break-words text-foreground/90">
                <code dangerouslySetInnerHTML={{ __html: highlightCode(file.content, file.language) }} />
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── File Details Panel ──────────────────────────────────────────────────

interface FileDetailsPanelProps {
  node: GraphNode;
  links: GraphLink[];
  nodes: GraphNode[];
  onSelectNode: (nodeId: string) => void;
  onClose: () => void;
}

function FileDetailsPanel({
  node,
  links,
  nodes,
  onSelectNode,
  onClose,
}: FileDetailsPanelProps) {
  const [stats, setStats] = useState({
    linesOfCode: 0,
    language: "",
    fileSize: 0,
    importCount: node.count,
    functionCount: 0,
    exportCount: 0,
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.read(node.path)
      .then((result) => {
        setStats({
          linesOfCode: result.lineCount,
          language: result.language,
          fileSize: 0,
          importCount: node.count,
          functionCount: countFunctions(result.content, result.language),
          exportCount: countExports(result.content, result.language),
        });

        api.browse(node.path.split("/").slice(0, -1).join("/"))
          .then((browseResult) => {
            const fileEntry = browseResult.entries.find((e: FileEntry) => e.path === node.path);
            if (fileEntry) {
              setStats(s => ({ ...s, fileSize: fileEntry.size || 0 }));
            }
          })
          .catch(() => {});

        setLoading(false);
      })
      .catch((err) => {
        console.error("Failed to load file:", err);
        setLoading(false);
      });
  }, [node]);

  const getNodeId = (ref: any): string => typeof ref === "string" ? ref : ref?.id || "";

  const dependsOn = links
    .filter(l => getNodeId(l.source) === node.id)
    .flatMap(l => {
      const dep = nodes.find(n => n.id === getNodeId(l.target));
      return dep ? [{ node: dep, symbols: l.symbols || [] }] : [];
    });

  const dependedOnBy = links
    .filter(l => getNodeId(l.target) === node.id)
    .flatMap(l => {
      const dep = nodes.find(n => n.id === getNodeId(l.source));
      return dep ? [{ node: dep, symbols: l.symbols || [] }] : [];
    });

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}M`;
  };

  return (
    <div className="flex flex-col h-full bg-card rounded-lg border border-border/50 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/30 bg-secondary flex-shrink-0">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-foreground truncate">{getFileName(node.path)}</div>
          <div className="text-[10px] text-muted-foreground truncate">{node.path}</div>
        </div>
        <button
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground transition-colors p-1 flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {loading ? (
          <div className="text-center text-muted-foreground py-8">Loading file details...</div>
        ) : (
          <>
            <div>
              <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Statistics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">Lines of Code</div>
                  <div className="text-sm font-medium text-foreground">{stats.linesOfCode}</div>
                </div>
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">Language</div>
                  <div className="text-sm font-medium text-foreground capitalize">{stats.language}</div>
                </div>
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">File Size</div>
                  <div className="text-sm font-medium text-foreground">{formatFileSize(stats.fileSize)}</div>
                </div>
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">Imports</div>
                  <div className="text-sm font-medium text-foreground">{stats.importCount}</div>
                </div>
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">Functions</div>
                  <div className="text-sm font-medium text-foreground">{stats.functionCount}</div>
                </div>
                <div className="bg-secondary rounded p-2">
                  <div className="text-[10px] text-muted-foreground">Exports/Classes</div>
                  <div className="text-sm font-medium text-foreground">{stats.exportCount}</div>
                </div>
              </div>
            </div>

            {node.exports && node.exports.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2">Exports</div>
                <div className="flex flex-wrap gap-1">
                  {node.exports.map(e => (
                    <span key={e} className="text-[10px] font-mono bg-secondary text-primary px-1.5 py-0.5 rounded">{e}</span>
                  ))}
                </div>
              </div>
            )}

            {dependsOn.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1">
                  <ArrowRightLeft className="w-3 h-3" />
                  Imports
                </div>
                <div className="space-y-1">
                  {dependsOn.map(({ node: dep, symbols }) => (
                    <button
                      key={dep.id}
                      onClick={() => onSelectNode(dep.id)}
                      className="w-full text-left rounded px-2 py-1.5 hover:bg-secondary transition-colors"
                    >
                      <div className="text-xs text-foreground/90 font-medium">{getFileName(dep.path)}</div>
                      {symbols.length > 0 && (
                        <div className="text-[10px] text-muted-foreground mt-0.5 font-mono">{symbols.join(", ")}</div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {dependedOnBy.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-2 flex items-center gap-1">
                  <Layers className="w-3 h-3" />
                  Imported By
                </div>
                <div className="space-y-1">
                  {dependedOnBy.map(({ node: dep, symbols }) => (
                    <button
                      key={dep.id}
                      onClick={() => onSelectNode(dep.id)}
                      className="w-full text-left rounded px-2 py-1.5 hover:bg-secondary transition-colors"
                    >
                      <div className="text-xs text-foreground/90 font-medium">{getFileName(dep.path)}</div>
                      {symbols.length > 0 && (
                        <div className="text-[10px] text-muted-foreground mt-0.5 font-mono">{symbols.join(", ")}</div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

// ── Graph View 3D helpers ───────────────────────────────────────────────

function langColor(lang: string | undefined): string {
  if (lang === "python")     return "hsl(142, 45%, 50%)";   // green
  if (lang === "typescript") return "hsl(210, 60%, 55%)";   // blue
  if (lang === "javascript") return "hsl(45,  65%, 55%)";   // yellow
  return "hsl(220, 20%, 50%)";                               // gray fallback
}

function langColorBright(lang: string | undefined): string {
  if (lang === "python")     return "rgba(34, 197, 94,  0.9)";
  if (lang === "typescript") return "rgba(99, 179, 255, 0.9)";
  if (lang === "javascript") return "rgba(234,210, 100, 0.9)";
  return "rgba(148, 163, 184, 0.85)";
}

function splitRgba(rgba: string): { color: string; alpha: number } {
  const m = rgba.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/);
  if (!m) return { color: rgba, alpha: 1 };
  return {
    color: `rgb(${m[1]}, ${m[2]}, ${m[3]})`,
    alpha: m[4] !== undefined ? parseFloat(m[4]) : 1,
  };
}

interface LangFilters {
  python: boolean;
  typescript: boolean;
  javascript: boolean;
}

// ── Graph View Component ────────────────────────────────────────────────

interface GraphViewProps {
  graphData: GraphData | null;
  loading: boolean;
  onSelectNode: (nodeId: string | null) => void;
  selectedNodeId: string | null;
}

function GraphView({ graphData, loading, onSelectNode, selectedNodeId }: GraphViewProps) {
  const fgRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const initialFitDone = useRef(false);
  const highlightNodesRef = useRef<Set<string>>(new Set());
  const lastClickRef = useRef<{ nodeId: string; ts: number } | null>(null);

  const [dimensions, setDimensions] = useState({ w: 800, h: 600 });
  const [langFilters, setLangFilters] = useState<LangFilters>({ python: true, typescript: true, javascript: true });
  const [activeNode, setActiveNode] = useState<GraphNode | null>(null);
  const [hoverNode, setHoverNode] = useState<GraphNode | null>(null);

  // Sync active node with external selection
  useEffect(() => {
    if (!graphData || !selectedNodeId) { setActiveNode(null); return; }
    const nd = graphData.nodes.find(n => n.id === selectedNodeId) || null;
    if (nd) setActiveNode(nd);
  }, [selectedNodeId, graphData]);

  // Resize tracking
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

  // Language-filtered data
  const filteredData = useMemo(() => {
    if (!graphData) return { nodes: [] as GraphNode[], links: [] as GraphLink[] };
    const filteredNodes = graphData.nodes.filter(n => {
      const lang = n.lang || "other";
      if (lang === "python"     && !langFilters.python)     return false;
      if (lang === "typescript" && !langFilters.typescript) return false;
      if (lang === "javascript" && !langFilters.javascript) return false;
      return true;
    });
    const nodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = graphData.links.filter(l => {
      const src = typeof l.source === "string" ? l.source : (l.source as any).id;
      const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
      return nodeIds.has(src) && nodeIds.has(tgt);
    });
    return { nodes: filteredNodes, links: filteredLinks };
  }, [graphData, langFilters]);

  // Highlight sets (active node or hover node fan-out)
  const { highlightNodes, highlightLinks } = useMemo(() => {
    const hn = new Set<string>();
    const hl = new Set<string>();
    const focusId = activeNode?.id ?? hoverNode?.id ?? null;
    if (focusId) {
      hn.add(focusId);
      for (const l of filteredData.links) {
        const src = typeof l.source === "string" ? l.source : (l.source as any).id;
        const tgt = typeof l.target === "string" ? l.target : (l.target as any).id;
        if (src === focusId) { hn.add(tgt); hl.add(`${src}\x00${tgt}`); }
        if (tgt === focusId) { hn.add(src); hl.add(`${src}\x00${tgt}`); }
      }
    }
    return { highlightNodes: hn, highlightLinks: hl };
  }, [activeNode, hoverNode, filteredData]);

  const hasDimming = !!(activeNode || hoverNode);

  // Keep ref in sync for the rAF label loop
  useEffect(() => { highlightNodesRef.current = highlightNodes; }, [highlightNodes]);

  // Fit view on first data load
  useEffect(() => {
    if (!filteredData.nodes.length || initialFitDone.current) return;
    const t = setTimeout(() => {
      if (fgRef.current) { fgRef.current.zoomToFit(1000, 50); initialFitDone.current = true; }
    }, 500);
    return () => clearTimeout(t);
  }, [filteredData]);

  // Force configuration
  const configureForces = useCallback(() => {
    const fg = fgRef.current;
    if (!fg) return;
    try {
      fg.d3Force("collide", forceCollide().radius(() => 9).strength(0.6).iterations(2));
      fg.d3Force("charge")?.strength(-120).distanceMax(500);
      fg.d3Force("link")?.distance(() => 55).strength(0.3);
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    if (!filteredData.nodes.length) return;
    configureForces();
  }, [filteredData, configureForces]);

  // Mesh opacity sync (highlight/dim)
  useEffect(() => {
    for (const n of filteredData.nodes as any[]) {
      const group = n.__threeObj as THREE.Object3D | undefined;
      const mesh = (group as any)?.__nodeMesh as THREE.Mesh | undefined;
      if (!mesh) continue;
      const isLit = highlightNodes.has(n.id);
      const opacity = hasDimming && !isLit ? 0.08 : 1;
      const mat = mesh.material as THREE.MeshPhongMaterial | undefined;
      if (mat && mat.opacity !== opacity) mat.opacity = opacity;
    }
  }, [highlightNodes, hasDimming, filteredData]);

  // Label visibility: highlight-based OR proximity-based (rAF loop)
  useEffect(() => {
    if (!filteredData.nodes.length) return;
    const PROXIMITY = 110;
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
      for (const n of filteredData.nodes as any[]) {
        const group = n.__threeObj as THREE.Object3D | undefined;
        const sprite = (group as any)?.__nodeLabel as THREE.Sprite | undefined;
        if (!sprite || !group) continue;
        if (anyLit) { sprite.visible = lit.has(n.id); continue; }
        group.getWorldPosition(tmp);
        sprite.visible = tmp.distanceTo(camera.position) < PROXIMITY;
      }
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [filteredData]);

  // Node THREE.js object
  const nodeThreeObject = useCallback((node: any) => {
    const n = node as GraphNode;
    const r = Math.max(2.5, Math.min(5, 2.5 + Math.sqrt(n.count || 0) * 0.35));
    const color = langColor(n.lang);
    const geometry = n.lang === "python"
      ? new THREE.OctahedronGeometry(r * 1.5, 0)
      : new THREE.SphereGeometry(r, 20, 16);
    const material = new THREE.MeshPhongMaterial({
      color, specular: 0x222222, shininess: 25, transparent: true, opacity: 1,
    });
    const mesh = new THREE.Mesh(geometry, material);

    const sprite = new SpriteText(getFileName(n.path));
    sprite.textHeight = 2.2;
    sprite.color = n.lang === "python" ? "#86efac" : n.lang === "typescript" ? "#93c5fd" : "#fde68a";
    sprite.backgroundColor = "rgba(15,23,42,0.65)";
    sprite.padding = 1.5;
    sprite.borderRadius = 2;
    sprite.position.set(0, r + 2.8, 0);
    sprite.visible = false;
    (sprite as any).raycast = () => {};

    const group = new THREE.Group();
    group.add(mesh);
    group.add(sprite);
    (group as any).__nodeMesh = mesh;
    (group as any).__nodeLabel = sprite;
    return group;
  }, []);

  // Link color
  const linkColorFn = useCallback((link: any) => {
    const src = typeof link.source === "string" ? link.source : link.source?.id;
    const tgt = typeof link.target === "string" ? link.target : link.target?.id;
    const key = `${src}\x00${tgt}`;
    const highlighted = highlightLinks.has(key);
    const dimmed = hasDimming && !highlighted;
    if (dimmed) return "rgba(71,85,105,0.08)";
    if (highlighted) {
      const srcNode = filteredData.nodes.find(n => n.id === src);
      return langColorBright(srcNode?.lang);
    }
    return "rgba(100,116,139,0.18)";
  }, [highlightLinks, hasDimming, filteredData]);

  // Flowing arrow particles on highlighted edges
  const linkParticlesFn = useCallback((link: any) => {
    const src = typeof link.source === "string" ? link.source : link.source?.id;
    const tgt = typeof link.target === "string" ? link.target : link.target?.id;
    return highlightLinks.has(`${src}\x00${tgt}`) ? 5 : 0;
  }, [highlightLinks]);

  const particleArrowFn = useCallback((link: any) => {
    const src = typeof link.source === "string" ? link.source : link.source?.id;
    const tgt = typeof link.target === "string" ? link.target : link.target?.id;
    if (!highlightLinks.has(`${src}\x00${tgt}`)) return undefined as any;
    const srcNode = filteredData.nodes.find(n => n.id === src);
    const { color, alpha } = splitRgba(langColorBright(srcNode?.lang));
    const geo = new THREE.ConeGeometry(0.7, 2.0, 6);
    geo.rotateX(Math.PI / 2);
    return new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: new THREE.Color(color), transparent: true, opacity: alpha,
    }));
  }, [highlightLinks, filteredData]);

  // Tooltip
  const nodeTooltip = useCallback((node: any) => {
    const n = node as GraphNode;
    const exStr = n.exports?.length
      ? `<br/><span style="color:#94a3b8;font-size:10px">exports: ${n.exports.slice(0, 3).join(", ")}${n.exports.length > 3 ? "…" : ""}</span>`
      : "";
    return `<b>${getFileName(n.path)}</b><br/><span style="color:#94a3b8;font-size:10px">${n.lang || "?"} · ${n.count} imports</span>${exStr}`;
  }, []);

  // Hover handler
  const handleNodeHover = useCallback((node: any) => {
    setHoverNode(node ? (node as GraphNode) : null);
  }, []);

  // Click handler
  const handleNodeClick = useCallback((node: any) => {
    const n = node as GraphNode;
    const now = Date.now();
    const last = lastClickRef.current;
    if (last && last.nodeId === n.id && now - last.ts < 300) {
      lastClickRef.current = null; return; // ignore double-click
    }
    lastClickRef.current = { nodeId: n.id, ts: now };
    if (activeNode?.id === n.id) {
      setActiveNode(null); onSelectNode(null);
    } else {
      setActiveNode(n); onSelectNode(n.id);
      if (fgRef.current && typeof n.x === "number") {
        const { x = 0, y = 0, z = 0 } = n;
        const dist = 120, mag = Math.hypot(x, y, z) || 1, k = 1 + dist / mag;
        fgRef.current.cameraPosition({ x: x*k, y: y*k, z: z*k }, { x, y, z }, 800);
      }
    }
  }, [activeNode, onSelectNode]);

  const handleBackgroundClick = useCallback(() => {
    setActiveNode(null); onSelectNode(null);
  }, [onSelectNode]);

  // Pin nodes when simulation settles
  const handleEngineStop = useCallback(() => {
    for (const n of filteredData.nodes as any[]) {
      if (typeof n.x === "number") n.fx = n.x;
      if (typeof n.y === "number") n.fy = n.y;
      if (typeof n.z === "number") n.fz = n.z;
    }
  }, [filteredData]);

  // Middle-click → pan (not zoom)
  useEffect(() => {
    if (!filteredData.nodes.length) return;
    const controls = fgRef.current?.controls?.();
    if (controls?.mouseButtons) controls.mouseButtons.MIDDLE = THREE.MOUSE.PAN;
  }, [filteredData]);

  // Custom lighting (like EntityGraph)
  const lights = useMemo(() => {
    const ambient = new THREE.AmbientLight(0x404060, 1.2);
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(150, 200, 100);
    const fill = new THREE.DirectionalLight(0x9fb8ff, 0.5);
    fill.position.set(-150, -50, -120);
    return [ambient, key, fill];
  }, []);

  if (loading) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-card rounded-lg border border-border/50 text-muted-foreground text-sm">
        Loading graph...
      </div>
    );
  }

  if (!graphData) {
    return (
      <div className="w-full h-full flex items-center justify-center bg-card rounded-lg border border-border/50 text-red-400 text-sm">
        Failed to load graph
      </div>
    );
  }

  return (
    <div ref={containerRef} className="w-full h-full relative bg-card rounded-lg border border-border/50 overflow-hidden">
      <div className="absolute inset-0">
        <ForceGraph3D
          ref={fgRef}
          width={dimensions.w}
          height={dimensions.h}
          graphData={filteredData as any}
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

      {/* Legend */}
      <div className="absolute top-2 left-2 z-10 bg-card/80 backdrop-blur-sm px-2 py-1.5 rounded text-[10px] text-muted-foreground space-y-1">
        <div className="font-semibold text-[9px] uppercase tracking-wider text-muted-foreground mb-0.5">Files</div>
        <div className="flex flex-col gap-0.5">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 flex-shrink-0 rotate-45" style={{ background: "hsl(142,45%,50%)" }} />
            python (◇)
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: "hsl(210,60%,55%)" }} />
            typescript
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ background: "hsl(45,65%,55%)" }} />
            javascript
          </span>
        </div>
        <div className="text-[9px] text-muted-foreground/70 mt-1">size = import count</div>
      </div>

      {/* Stats */}
      <div className="absolute top-2 right-2 bg-secondary/90 px-3 py-2 rounded text-[11px] text-muted-foreground">
        {filteredData.nodes.length} nodes · {filteredData.links.length} links
      </div>

      {/* Controls: language filters + reset/fit */}
      <div className="absolute bottom-2 right-2 z-10 flex items-center gap-1">
        <div className="flex items-center gap-2 mr-2 bg-card/80 backdrop-blur-sm px-2 py-1 rounded text-[10px]">
          {(["python", "typescript", "javascript"] as const).map((lang) => (
            <label key={lang} className="flex items-center gap-1 cursor-pointer select-none text-muted-foreground hover:text-foreground">
              <input
                type="checkbox"
                checked={langFilters[lang]}
                onChange={() => setLangFilters(prev => ({ ...prev, [lang]: !prev[lang] }))}
                className="w-2.5 h-2.5 accent-primary cursor-pointer"
              />
              <span>{lang}</span>
            </label>
          ))}
        </div>
        <button
          onClick={() => { setActiveNode(null); onSelectNode(null); }}
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
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────

const BROWSER_TABS = [
  { id: "lloyd", label: "lloyd", path: "/home/alansrobotlab/lloyd", scope: "py" },
  { id: "lloyd-web", label: "web", path: "/home/alansrobotlab/lloyd/web/src", scope: "ts" },
] as const;
type BrowserTab = typeof BROWSER_TABS[number]["id"];

export default function ArchitecturePage() {
  const [browserTab, setBrowserTab] = useState<BrowserTab>("lloyd");
  const currentTab = BROWSER_TABS.find(t => t.id === browserTab)!;
  const currentPath = currentTab.path;
  const currentScope = currentTab.scope;
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileContent | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);

  // Refetch when the tab (and therefore the graph scope) changes. Clear the
  // current selection since node ids from the previous scope won't resolve.
  useEffect(() => {
    setGraphLoading(true);
    setGraphData(null);
    setSelectedNodeId(null);
    api.fetchGraph(currentScope)
      .then(data => {
        setGraphData(data);
        setGraphLoading(false);
      })
      .catch(err => {
        console.error("Failed to load graph:", err);
        setGraphLoading(false);
      });
  }, [currentScope]);

  const handleOpenFile = useCallback(async (filePath: string) => {
    setLoadingFile(true);
    try {
      const result = await api.read(filePath);
      setSelectedFile(result);
    } catch (err) {
      console.error("Failed to read file:", err);
      setSelectedFile(null);
    } finally {
      setLoadingFile(false);
    }
  }, []);

  const handleCloseFile = useCallback(() => {
    setSelectedFile(null);
  }, []);

  const handleSelectNode = useCallback((nodeId: string | null) => {
    setSelectedNodeId(nodeId);
  }, []);

  const selectedNode = selectedNodeId && graphData
    ? graphData.nodes.find(n => n.id === selectedNodeId) || null
    : null;

  return (
    <div className="flex flex-col h-full min-h-0 gap-3 p-6">
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className="flex items-center gap-2">
          <Code2 className="w-5 h-5 text-primary" />
          <h1 className="text-lg font-bold text-foreground">Architecture</h1>
        </div>
        <div className="text-[11px] text-muted-foreground">
          Browse source: <code className="bg-secondary px-1 rounded">~/lloyd/</code> and <code className="bg-secondary px-1 rounded">~/lloyd/web/src/</code>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left panel: File tree */}
        <div className="w-64 flex-shrink-0 flex flex-col min-h-0 bg-card rounded-lg border border-border/50 overflow-hidden">
          <div className="flex border-b border-border/30 bg-secondary flex-shrink-0">
            {BROWSER_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setBrowserTab(tab.id)}
                className={`flex-1 px-2 py-2 text-[11px] font-medium transition-colors ${
                  browserTab === tab.id
                    ? "text-primary border-b-2 border-primary bg-card/50"
                    : "text-muted-foreground hover:text-foreground/90"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto py-2">
            {loadingFile ? (
              <div className="text-[11px] text-muted-foreground px-2 py-4">Loading file...</div>
            ) : (
              <FileTree path={currentPath} onOpenFile={handleOpenFile} />
            )}
          </div>
        </div>

        {/* Center panel: Dependency graph */}
        <div className="flex-1 min-w-0 min-h-0">
          <GraphView
            graphData={graphData}
            loading={graphLoading}
            onSelectNode={handleSelectNode}
            selectedNodeId={selectedNodeId}
          />
        </div>

        {/* Right panel: File details (conditional) */}
        {selectedNode && (
          <div className="w-80 flex-shrink-0 flex flex-col min-h-0">
            <FileDetailsPanel
              node={selectedNode}
              links={graphData?.links || []}
              nodes={graphData?.nodes || []}
              onSelectNode={handleSelectNode}
              onClose={() => handleSelectNode(null)}
            />
          </div>
        )}
      </div>

      {selectedFile && (
        <FileViewerModal file={selectedFile} onClose={handleCloseFile} />
      )}
    </div>
  );
}

// ── API helper ──────────────────────────────────────────────────────────

const api = {
  browse: (path: string) =>
    fetch(`/api/architecture/browse?path=${encodeURIComponent(path)}`).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }),
  read: (path: string) =>
    fetch(`/api/architecture/read?path=${encodeURIComponent(path)}`).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }),
  fetchGraph: (scope?: string) =>
    fetch(`/api/architecture/graph${scope ? `?scope=${encodeURIComponent(scope)}` : ""}`).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }),
};
