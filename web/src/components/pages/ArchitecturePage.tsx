import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import ForceGraph2D from "react-force-graph-2d";
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
  x?: number;
  y?: number;
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

function getColorFromPath(path: string): string {
  const parentDir = path.split("/").slice(0, -1).join("/");
  let hash = 0;
  for (let i = 0; i < parentDir.length; i++) {
    hash = ((hash << 5) - hash + parentDir.charCodeAt(i)) | 0;
  }
  const hue = ((hash % 360) + 360) % 360;
  return `hsl(${hue}, 60%, 55%)`;
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
            className="w-full flex items-center gap-1.5 py-0.5 text-[11px] text-slate-300 hover:bg-surface-2 rounded transition-colors"
            style={{ paddingLeft: `${indent + 4}px` }}
          >
            {node.expanded ? (
              <ChevronDown className="w-3 h-3 text-slate-500 flex-shrink-0" />
            ) : (
              <ChevronRight className="w-3 h-3 text-slate-500 flex-shrink-0" />
            )}
            {node.expanded ? (
              <FolderOpen className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
            ) : (
              <Folder className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
            )}
            <span className="truncate">{node.name}</span>
            {node.children != null && (
              <span className="text-[10px] text-slate-600 ml-auto pr-1">{node.children}</span>
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
        className="w-full flex items-center gap-1.5 py-0.5 text-[11px] text-slate-400 hover:text-slate-200 hover:bg-surface-2 rounded transition-colors"
        style={{ paddingLeft: `${indent + 4 + 15}px` }}
      >
        <FileText className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
        <span className="truncate">{node.name}</span>
        {node.size != null && (
          <span className="text-[10px] text-slate-600 ml-auto pr-1">
            {node.size < 1024 ? `${node.size}B` : `${(node.size / 1024).toFixed(1)}K`}
          </span>
        )}
      </button>
    );
  };

  if (loading) {
    return <div className="text-[11px] text-slate-500 px-2 py-4">Loading...</div>;
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
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-8">
      <div className="bg-surface-1 rounded-lg border border-surface-3/50 overflow-hidden flex flex-col max-w-5xl w-full max-h-[90vh]">
        <div className="flex items-center justify-between px-4 py-2 border-b border-surface-3/30 bg-surface-2 flex-shrink-0">
          <div className="flex items-center gap-2 min-w-0">
            <Code2 className="w-4 h-4 text-brand-400 flex-shrink-0" />
            <span className="text-sm font-medium text-slate-200 truncate">{file.path}</span>
          </div>
          <div className="flex items-center gap-3 text-[11px] text-slate-500 flex-shrink-0">
            <span>{file.language}</span>
            <span>{file.lineCount} lines</span>
            <button
              onClick={onClose}
              className="text-slate-400 hover:text-slate-200 transition-colors p-1"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto font-mono text-xs">
          <div className="flex">
            <div className="w-12 flex-shrink-0 bg-surface-2/50 text-slate-600 text-right pr-3 py-2 select-none border-r border-surface-3/20">
              {lines.map((_, i) => (
                <div key={i} className="leading-5">{i + 1}</div>
              ))}
            </div>
            <div className="flex-1 p-2 overflow-x-auto">
              <pre className="leading-5 whitespace-pre-wrap break-words text-slate-300">
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
    <div className="flex flex-col h-full bg-surface-1 rounded-lg border border-surface-3/50 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-surface-3/30 bg-surface-2 flex-shrink-0">
        <div className="min-w-0">
          <div className="text-sm font-semibold text-slate-200 truncate">{getFileName(node.path)}</div>
          <div className="text-[10px] text-slate-500 truncate">{node.path}</div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 transition-colors p-1 flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        {loading ? (
          <div className="text-center text-slate-500 py-8">Loading file details...</div>
        ) : (
          <>
            <div>
              <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Statistics</div>
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">Lines of Code</div>
                  <div className="text-sm font-medium text-slate-200">{stats.linesOfCode}</div>
                </div>
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">Language</div>
                  <div className="text-sm font-medium text-slate-200 capitalize">{stats.language}</div>
                </div>
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">File Size</div>
                  <div className="text-sm font-medium text-slate-200">{formatFileSize(stats.fileSize)}</div>
                </div>
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">Imports</div>
                  <div className="text-sm font-medium text-slate-200">{stats.importCount}</div>
                </div>
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">Functions</div>
                  <div className="text-sm font-medium text-slate-200">{stats.functionCount}</div>
                </div>
                <div className="bg-surface-2 rounded p-2">
                  <div className="text-[10px] text-slate-500">Exports/Classes</div>
                  <div className="text-sm font-medium text-slate-200">{stats.exportCount}</div>
                </div>
              </div>
            </div>

            {node.exports && node.exports.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Exports</div>
                <div className="flex flex-wrap gap-1">
                  {node.exports.map(e => (
                    <span key={e} className="text-[10px] font-mono bg-surface-2 text-brand-400 px-1.5 py-0.5 rounded">{e}</span>
                  ))}
                </div>
              </div>
            )}

            {dependsOn.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                  <ArrowRightLeft className="w-3 h-3" />
                  Imports
                </div>
                <div className="space-y-1">
                  {dependsOn.map(({ node: dep, symbols }) => (
                    <button
                      key={dep.id}
                      onClick={() => onSelectNode(dep.id)}
                      className="w-full text-left rounded px-2 py-1.5 hover:bg-surface-2 transition-colors"
                    >
                      <div className="text-xs text-slate-300 font-medium">{getFileName(dep.path)}</div>
                      {symbols.length > 0 && (
                        <div className="text-[10px] text-slate-500 mt-0.5 font-mono">{symbols.join(", ")}</div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {dependedOnBy.length > 0 && (
              <div>
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-1">
                  <Layers className="w-3 h-3" />
                  Imported By
                </div>
                <div className="space-y-1">
                  {dependedOnBy.map(({ node: dep, symbols }) => (
                    <button
                      key={dep.id}
                      onClick={() => onSelectNode(dep.id)}
                      className="w-full text-left rounded px-2 py-1.5 hover:bg-surface-2 transition-colors"
                    >
                      <div className="text-xs text-slate-300 font-medium">{getFileName(dep.path)}</div>
                      {symbols.length > 0 && (
                        <div className="text-[10px] text-slate-500 mt-0.5 font-mono">{symbols.join(", ")}</div>
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

// ── Graph View Component ────────────────────────────────────────────────

interface GraphViewProps {
  graphData: GraphData | null;
  loading: boolean;
  onSelectNode: (nodeId: string | null) => void;
}

function GraphView({
  graphData,
  loading,
  onSelectNode,
}: GraphViewProps) {
  const nodeColor = (node: GraphNode) => {
    const parentDir = node.path.split("/").slice(0, -1).join("/");
    let hash = 0;
    for (let i = 0; i < parentDir.length; i++) {
      hash = ((hash << 5) - hash + parentDir.charCodeAt(i)) | 0;
    }
    const hue = ((hash % 60) + 60) % 60; // narrow hue range per-language
    if (node.lang === "typescript") return `hsl(${200 + hue}, 60%, 55%)`; // blues
    return `hsl(${100 + hue}, 55%, 50%)`; // greens for python
  };
  const stableData = useMemo(
    () => graphData ? { nodes: graphData.nodes, links: graphData.links } : { nodes: [], links: [] },
    [graphData]
  );

  return (
    <div className="flex-1 relative bg-surface-1 rounded-lg border border-surface-3/50 overflow-hidden">
      {loading ? (
        <div className="absolute inset-0 flex items-center justify-center text-slate-500">
          Loading graph...
        </div>
      ) : graphData ? (
        <>
          <ForceGraph2D
            graphData={stableData}
            nodeLabel={(node: any) => getFileName(node.path)}
            nodeColor={nodeColor}
            nodeRelSize={3}
            nodeVal="count"
            linkDirectionalArrowLength={5}
            linkCurvature={0.3}
            linkColor={() => "rgba(148, 163, 184, 0.4)"}
            linkWidth={1}
            backgroundColor="transparent"
            nodeCanvasObjectMode={() => "after"}
            nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
              const label = getFileName(node.path || node.id);
              const fontSize = Math.max(10 / globalScale, 2);
              ctx.font = `${fontSize}px sans-serif`;
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillStyle = "rgba(203, 213, 225, 0.8)";
              ctx.fillText(label, node.x, node.y + 5);
            }}
            onNodeClick={(node: GraphNode) => {
              onSelectNode(node.id);
            }}
            minZoom={0.2}
            maxZoom={4}
            warmupTicks={100}
            cooldownTicks={200}
            cooldownTime={2000}
            d3AlphaMin={0.001}
            d3VelocityDecay={0.3}
          />
          <div className="absolute top-2 right-2 bg-surface-2/90 px-3 py-2 rounded text-[11px] text-slate-500">
            {graphData.totalNodes} nodes · {graphData.totalLinks} links
          </div>
        </>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center text-red-400">
          Failed to load graph
        </div>
      )}
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────

const BROWSER_TABS = [
  { id: "lloyd", label: "lloyd", path: "/home/alansrobotlab/lloyd" },
  { id: "lloyd-web", label: "web", path: "/home/alansrobotlab/lloyd/web/src" },
] as const;
type BrowserTab = typeof BROWSER_TABS[number]["id"];

export default function ArchitecturePage() {
  const [browserTab, setBrowserTab] = useState<BrowserTab>("lloyd");
  const currentPath = BROWSER_TABS.find(t => t.id === browserTab)!.path;
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [graphLoading, setGraphLoading] = useState(true);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<FileContent | null>(null);
  const [loadingFile, setLoadingFile] = useState(false);

  useEffect(() => {
    setGraphLoading(true);
    api.fetchGraph()
      .then(data => {
        setGraphData(data);
        setGraphLoading(false);
      })
      .catch(err => {
        console.error("Failed to load graph:", err);
        setGraphLoading(false);
      });
  }, []);

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
          <Code2 className="w-5 h-5 text-brand-400" />
          <h1 className="text-lg font-bold text-slate-200">Architecture</h1>
        </div>
        <div className="text-[11px] text-slate-500">
          Browse source: <code className="bg-surface-2 px-1 rounded">~/lloyd/</code> and <code className="bg-surface-2 px-1 rounded">~/lloyd/web/src/</code>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left panel: File tree */}
        <div className="w-64 flex-shrink-0 flex flex-col min-h-0 bg-surface-1 rounded-lg border border-surface-3/50 overflow-hidden">
          <div className="flex border-b border-surface-3/30 bg-surface-2 flex-shrink-0">
            {BROWSER_TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setBrowserTab(tab.id)}
                className={`flex-1 px-2 py-2 text-[11px] font-medium transition-colors ${
                  browserTab === tab.id
                    ? "text-brand-400 border-b-2 border-brand-400 bg-surface-1/50"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto py-2">
            {loadingFile ? (
              <div className="text-[11px] text-slate-500 px-2 py-4">Loading file...</div>
            ) : (
              <FileTree path={currentPath} onOpenFile={handleOpenFile} />
            )}
          </div>
        </div>

        {/* Center panel: Dependency graph */}
        <GraphView
          graphData={graphData}
          loading={graphLoading}
          onSelectNode={handleSelectNode}
        />

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
  fetchGraph: () =>
    fetch(`/api/architecture/graph`).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    }),
};
