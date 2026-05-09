import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { useReportMcFocus, usePendingFocusFor } from "../../contexts/McUiContext";
import {
  Search,
  FileText,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Tag,
  User,
  X,
  Pencil,
  Check,
  Network,
} from "lucide-react";
import {
  api,
  type MemoryStats,
  type MemorySearchResult,
  type MemoryReadResult,
  type EntitySummary,
  type EntityDetailData,
  type EntityFact,
  type EntityGraphNode,
  type EntityGraphData,
} from "../../api";
import { Streamdown } from "streamdown";
import EntityGraph, { type EntityGraphProps } from "../EntityGraph";
import { categoryOf, CATEGORY_CHIP_CLASS } from "../../lib/edgeCategories";

// -- Types --

type SidebarTab = "entities" | "explorer";

// Re-export the GNode type from EntityGraph (same shape)
interface GNode extends EntityGraphNode {
  x?: number;
  y?: number;
  degree?: number;
}

const TYPE_COLORS: Record<string, string> = {
  hub: "bg-amber-400/10 text-amber-400",
  notes: "bg-slate-400/10 text-muted-foreground",
  "project-notes": "bg-sky-400/10 text-sky-400",
  "work-notes": "bg-indigo-400/10 text-indigo-400",
  talk: "bg-emerald-400/10 text-emerald-400",
  reference: "bg-purple-400/10 text-purple-400",
};

const CATEGORY_COLORS: Record<string, string> = {
  profile: "text-sky-400",
  preference: "text-primary",
  event: "text-emerald-400",
  state: "text-amber-400",
  relationship: "text-purple-400",
  people: "text-pink-400",
  project: "text-indigo-400",
};

// -- NodeQuickInfo Component --

function NodeQuickInfo({
  node,
  graphData,
}: {
  node: GNode;
  graphData: { nodes: EntityGraphNode[]; edges: { source: string | EntityGraphNode; target: string | EntityGraphNode; type: string }[] } | null;
}) {
  const getNodeId = (n: string | EntityGraphNode) => typeof n === "string" ? n : n.id;

  const connectedNodes = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges
      .filter((e) => getNodeId(e.source) === node.id || getNodeId(e.target) === node.id)
      .map((e) => {
        const otherId = getNodeId(e.source) === node.id ? getNodeId(e.target) : getNodeId(e.source);
        const otherNode = graphData.nodes.find((n) => n.id === otherId);
        return { id: otherId, label: otherNode?.label || otherId.split("/").pop()?.replace(/\.md$/, "") || otherId, type: e.type };
      })
      .slice(0, 10);
  }, [node, graphData]);

  const connectionCount = useMemo(() => {
    if (!graphData) return 0;
    return graphData.edges.filter((e) => getNodeId(e.source) === node.id || getNodeId(e.target) === node.id).length;
  }, [node, graphData]);

  const vaultSection = useMemo(() => {
    if (node.type === "entity") return "entity";
    const id = node.id;
    if (id.startsWith("memory/")) return "memory";
    if (id.startsWith("projects/")) return "projects";
    if (id.startsWith("knowledge/")) return "knowledge";
    if (id.startsWith("agents/")) return "agents";
    return "unknown";
  }, [node]);

  if (node.type === "entity") {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 border-b border-border/30 pb-2">
          <span className="inline-block w-2 h-2 rotate-45 flex-shrink-0" style={{background: "#F59E0B"}} />
          <span className="text-xs font-semibold text-amber-400">{node.label}</span>
          <span className="ml-auto text-[10px] text-muted-foreground">entity</span>
        </div>
        <div className="text-[10px] text-muted-foreground">
          <span className="text-muted-foreground">{node.factCount || 0}</span> facts
          {" · "}
          <span className="text-muted-foreground">{connectionCount}</span> connections
        </div>
        {connectedNodes.length > 0 && (
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Connected docs</div>
            <div className="space-y-0.5">
              {connectedNodes.map((n, i) => (
                <div key={i} className="text-[10px] text-muted-foreground truncate flex items-center gap-1">
                  <FileText className="w-2.5 h-2.5 flex-shrink-0 opacity-40" />
                  <span className="truncate">{n.label}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 border-b border-border/30 pb-2">
        <FileText className="w-3 h-3 text-muted-foreground flex-shrink-0" />
        <span className="text-xs font-semibold text-foreground truncate">{node.label}</span>
      </div>
      <div className="font-mono text-[9px] text-muted-foreground break-all leading-relaxed">{node.id}</div>
      <div className="flex flex-wrap gap-1.5 text-[10px]">
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{vaultSection}</span>
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{connectionCount} links</span>
        {node.degree !== undefined && (
          <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">deg {node.degree}</span>
        )}
      </div>
      {connectedNodes.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Connected</div>
          <div className="space-y-0.5">
            {connectedNodes.map((n, i) => (
              <div key={i} className="text-[10px] text-muted-foreground truncate flex items-center gap-1">
                <span className={`flex-shrink-0 text-[9px] px-1 py-0.5 rounded ${CATEGORY_CHIP_CLASS[categoryOf(n.type)]}`}>{n.type}</span>
                <span className="truncate">{n.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// -- DocNodeDetail Component (single-click sticky) --

function DocNodeDetail({
  doc,
  graphData,
}: {
  doc: MemoryReadResult;
  graphData: { nodes: EntityGraphNode[]; edges: { source: string | EntityGraphNode; target: string | EntityGraphNode; type: string }[] } | null;
}) {
  const fm = doc.frontmatter;
  const typeClass = TYPE_COLORS[fm.type] || TYPE_COLORS.notes;

  const getNodeId = (n: string | EntityGraphNode) => typeof n === "string" ? n : n.id;

  const connectionCount = useMemo(() => {
    if (!graphData) return 0;
    return graphData.edges.filter((e) => getNodeId(e.source) === doc.path || getNodeId(e.target) === doc.path).length;
  }, [doc.path, graphData]);

  const connectedDocs = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges
      .filter((e) => getNodeId(e.source) === doc.path || getNodeId(e.target) === doc.path)
      .map((e) => {
        const otherId = getNodeId(e.source) === doc.path ? getNodeId(e.target) : getNodeId(e.source);
        const otherNode = graphData.nodes.find((n) => n.id === otherId);
        return { id: otherId, label: otherNode?.label || otherId.split("/").pop()?.replace(/\.md$/, "") || otherId, type: e.type };
      })
      .slice(0, 15);
  }, [doc.path, graphData]);

  const sizeKb = useMemo(() => {
    const bytes = new Blob([doc.content]).size;
    return bytes < 1024 ? `${bytes}B` : `${(bytes / 1024).toFixed(1)}K`;
  }, [doc.content]);

  const vaultSection = useMemo(() => {
    const id = doc.path;
    if (id.startsWith("memory/")) return "memory";
    if (id.startsWith("projects/")) return "projects";
    if (id.startsWith("knowledge/")) return "knowledge";
    if (id.startsWith("agents/")) return "agents";
    return "unknown";
  }, [doc.path]);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 border-b border-border/30 pb-2">
        <FileText className="w-3.5 h-3.5 text-primary flex-shrink-0" />
        <span className="text-xs font-semibold text-foreground truncate">{fm.title || doc.path}</span>
      </div>
      <div className="font-mono text-[9px] text-muted-foreground break-all leading-relaxed">{doc.path}</div>
      <div className="flex flex-wrap gap-1.5 text-[10px]">
        {fm.type && <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${typeClass}`}>{fm.type}</span>}
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{vaultSection}</span>
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{sizeKb}</span>
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{doc.lineCount} lines</span>
        <span className="bg-secondary px-1.5 py-0.5 rounded text-muted-foreground">{connectionCount} links</span>
      </div>
      {fm.summary && (
        <div className="text-[11px] text-muted-foreground italic leading-relaxed bg-secondary/40 rounded px-2 py-1">
          {fm.summary}
        </div>
      )}
      {Array.isArray(fm.tags) && fm.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {fm.tags.map((tag: string) => (
            <span key={tag} className="text-[10px] text-muted-foreground bg-secondary px-1.5 py-0.5 rounded">#{tag}</span>
          ))}
        </div>
      )}
      {connectedDocs.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">Connected docs</div>
          <div className="space-y-0.5">
            {connectedDocs.map((n, i) => (
              <div key={i} className="text-[10px] text-muted-foreground truncate flex items-center gap-1">
                <span className={`flex-shrink-0 text-[9px] px-1 py-0.5 rounded ${CATEGORY_CHIP_CLASS[categoryOf(n.type)]}`}>{n.type}</span>
                <span className="truncate">{n.label}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// -- Entity Sidebar --

function EntitySidebar({
  entities,
  activeEntity,
  onSelectEntity,
}: {
  entities: EntitySummary[];
  activeEntity: string | null;
  onSelectEntity: (name: string | null) => void;
}) {
  const activeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (activeEntity && activeRef.current) {
      activeRef.current.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [activeEntity]);

  return (
    <div className="space-y-0.5">
      {activeEntity && (
        <div className="flex items-center px-1 mb-1">
          <button
            onClick={() => onSelectEntity(null)}
            className="ml-auto text-[10px] text-primary hover:text-primary"
          >
            clear
          </button>
        </div>
      )}
      {entities.map(({ name, factCount }) => (
        <button
          key={name}
          ref={activeEntity === name ? activeRef : undefined}
          onClick={() => onSelectEntity(activeEntity === name ? null : name)}
          className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] transition-colors ${
            activeEntity === name
              ? "bg-primary/15 text-primary"
              : "text-muted-foreground hover:text-foreground hover:bg-secondary"
          }`}
        >
          <User className="w-2.5 h-2.5 flex-shrink-0 opacity-50" />
          <span className="flex-1 text-left truncate">{name}</span>
          <span className="text-[10px] text-muted-foreground flex-shrink-0">{factCount}</span>
        </button>
      ))}
    </div>
  );
}

// -- Entity Detail Panel --

function EntityDetailPanel({ detail }: { detail: EntityDetailData }) {
  // Group facts by category
  const byCategory = useMemo(() => {
    const map = new Map<string, EntityFact[]>();
    for (const f of detail.facts) {
      const cat = f.category || "general";
      if (!map.has(cat)) map.set(cat, []);
      map.get(cat)!.push(f);
    }
    return map;
  }, [detail.facts]);

  const categories = Array.from(byCategory.keys()).sort();

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 border-b border-border/30 pb-2">
        <User className="w-3.5 h-3.5 text-primary flex-shrink-0" />
        <span className="text-xs font-semibold text-foreground">{detail.name}</span>
        <span className="ml-auto text-[10px] text-muted-foreground">{detail.facts.length} facts</span>
      </div>

      {detail.definition && (
        <div className="text-[11px] text-muted-foreground italic leading-snug">
          {detail.definition}
        </div>
      )}

      {detail.summary && (
        <div className="text-[11px] text-foreground/90 leading-relaxed whitespace-pre-wrap bg-secondary/30 rounded px-2 py-1.5">
          {detail.summary}
        </div>
      )}

      {categories.map((cat) => {
        const facts = byCategory.get(cat)!;
        const colorClass = CATEGORY_COLORS[cat] || "text-muted-foreground";
        return (
          <div key={cat}>
            <div className={`text-[10px] font-semibold uppercase tracking-wider mb-1 ${colorClass}`}>
              {cat}
            </div>
            <div className="space-y-1">
              {facts.map((f, i) => (
                <div
                  key={f.id || i}
                  className="text-[11px] text-foreground/90 leading-relaxed bg-secondary/40 rounded px-2 py-1"
                >
                  {f.fact}
                  {f.confidence < 0.8 && (
                    <span className="ml-1 text-[9px] text-muted-foreground opacity-70">
                      ({Math.round(f.confidence * 100)}%)
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        );
      })}

      {detail.relationships.length > 0 && (
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1 text-muted-foreground">
            Related docs
          </div>
          <div className="space-y-0.5">
            {detail.relationships.slice(0, 10).map((r, i) => (
              <div key={i} className="text-[10px] text-muted-foreground truncate flex items-center gap-1">
                <span className={`flex-shrink-0 text-[9px] px-1 py-0.5 rounded ${CATEGORY_CHIP_CLASS[categoryOf(r.type)]}`}>
                  {r.type}
                </span>
                <span className="truncate">{r.target.split("/").pop()?.replace(/\.md$/, "")}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// -- Vault Explorer (tree view) --

interface TreeNode {
  name: string;
  path: string;
  type: "file" | "dir";
  size?: number;
  title?: string;
  children?: number;
  entries?: TreeNode[];
  loaded?: boolean;
  expanded?: boolean;
}

function VaultExplorer({ onOpenFile }: { onOpenFile: (path: string) => void }) {
  const [tree, setTree] = useState<TreeNode[]>([]);
  const treeRef = useRef<TreeNode[]>([]);
  treeRef.current = tree;

  useEffect(() => {
    api.memoryBrowse("").then(async (root) => {
      const nodes: TreeNode[] = root.entries.map((e) => ({
        ...e,
        path: e.name,
        loaded: false,
        expanded: false,
      }));

      setTree(nodes);
    }).catch(console.error);
  }, []);

  const toggleDir = useCallback(async (path: string) => {
    const updateNodes = async (nodes: TreeNode[]): Promise<TreeNode[]> => {
      const result: TreeNode[] = [];
      for (const node of nodes) {
        if (node.path === path && node.type === "dir") {
          if (!node.loaded) {
            try {
              const sub = await api.memoryBrowse(path);
              result.push({
                ...node,
                expanded: true,
                loaded: true,
                entries: sub.entries.map((e) => ({
                  ...e,
                  path: `${path}/${e.name}`,
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
        <span className="truncate">{node.title || node.name}</span>
        {node.size != null && (
          <span className="text-[10px] text-muted-foreground/70 ml-auto pr-1">
            {node.size < 1024 ? `${node.size}B` : `${(node.size / 1024).toFixed(1)}K`}
          </span>
        )}
      </button>
    );
  };

  if (tree.length === 0) {
    return <div className="text-[11px] text-muted-foreground px-2 py-4">Loading...</div>;
  }

  return <div className="space-y-0">{tree.map((node) => renderNode(node, 0))}</div>;
}

// -- Search Results (right panel listing) --

function SearchResults({
  results,
  query,
  onOpenFile,
}: {
  results: MemorySearchResult;
  query: string;
  onOpenFile: (path: string) => void;
}) {
  if (results.results.length === 0) {
    return (
      <div className="text-center py-12 text-muted-foreground">
        <Search className="w-6 h-6 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No results for "{query}"</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-[10px] text-muted-foreground px-1">
        {results.results.length} results for "{query}"
      </div>
      {results.results.map((r) => (
        <button
          key={r.path}
          onClick={() => onOpenFile(r.path)}
          className="w-full text-left bg-card rounded-lg p-2.5 border border-border/50 hover:border-primary/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <FileText className="w-3.5 h-3.5 text-primary flex-shrink-0" />
            <span className="text-xs font-medium text-foreground truncate">
              {r.title || r.path}
            </span>
          </div>
          {(r.summary || r.snippet) && (
            <p className="text-[11px] text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
              {r.summary || r.snippet}
            </p>
          )}
        </button>
      ))}
    </div>
  );
}

// -- Document Modal --

function DocumentModal({
  doc,
  onClose,
  onSaved,
}: {
  doc: MemoryReadResult;
  onClose: () => void;
  onSaved: (updated: MemoryReadResult) => void;
}) {
  const fm = doc.frontmatter;
  const typeClass = TYPE_COLORS[fm.type] || TYPE_COLORS.notes;

  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState(doc.content);
  const [editTags, setEditTags] = useState((fm.tags || []).join(", "));
  const [editSummary, setEditSummary] = useState(fm.summary || "");
  const [saving, setSaving] = useState(false);

  const handleEdit = () => {
    setEditContent(doc.content);
    setEditTags((fm.tags || []).join(", "));
    setEditSummary(fm.summary || "");
    setEditing(true);
  };

  const handleCancel = () => {
    setEditing(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      const tags = editTags.split(",").map((t: string) => t.trim()).filter(Boolean);
      await api.memorySave(doc.path, editContent, { tags, summary: editSummary });
      const updated = await api.memoryRead(doc.path);
      onSaved(updated);
      setEditing(false);
    } catch (err) {
      console.error("Save failed:", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget && !editing) onClose(); }}
    >
      <div className="bg-card border border-border/50 rounded-xl shadow-2xl w-[80%] h-[80%] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-border/30 flex-shrink-0">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-foreground truncate">
              {fm.title || doc.path}
            </div>
            <div className="text-[10px] text-muted-foreground font-mono">{doc.path}</div>
          </div>
          <div className="flex items-center gap-1.5">
            {editing ? (
              <>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md bg-primary text-white hover:bg-primary disabled:opacity-50 transition-colors"
                >
                  <Check className="w-3.5 h-3.5" />
                  {saving ? "Saving..." : "Save"}
                </button>
                <button
                  onClick={handleCancel}
                  disabled={saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary disabled:opacity-50 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={handleEdit}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary transition-colors"
                >
                  <Pencil className="w-3.5 h-3.5" />
                  Edit
                </button>
                <button
                  onClick={onClose}
                  className="text-muted-foreground hover:text-foreground transition-colors p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Frontmatter badges / editable tags */}
        <div className="flex items-center gap-2 px-5 py-2 flex-wrap flex-shrink-0 border-b border-border/20">
          {fm.type && (
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded ${typeClass}`}>
              {fm.type}
            </span>
          )}
          {fm.status && (
            <span className="text-[10px] text-muted-foreground bg-secondary px-2 py-0.5 rounded">
              {fm.status}
            </span>
          )}
          {editing ? (
            <div className="flex items-center gap-1.5 flex-1 min-w-0">
              <span className="text-[10px] text-muted-foreground flex-shrink-0">tags:</span>
              <input
                value={editTags}
                onChange={(e) => setEditTags(e.target.value)}
                placeholder="tag1, tag2, tag3"
                className="flex-1 bg-background text-[11px] text-foreground rounded px-2 py-0.5 border border-border/50 outline-none focus:border-primary/50 font-mono"
              />
            </div>
          ) : (
            Array.isArray(fm.tags) &&
              fm.tags.map((tag: string) => (
                <span
                  key={tag}
                  className="text-[10px] text-muted-foreground bg-secondary px-1.5 py-0.5 rounded"
                >
                  #{tag}
                </span>
              ))
          )}
          {!editing && <span className="text-[10px] text-muted-foreground ml-auto">{doc.lineCount} lines</span>}
        </div>

        {/* Summary */}
        {editing ? (
          <div className="flex items-center gap-1.5 px-5 py-2 flex-shrink-0 border-b border-border/20">
            <span className="text-[10px] text-muted-foreground flex-shrink-0">summary:</span>
            <input
              value={editSummary}
              onChange={(e) => setEditSummary(e.target.value)}
              placeholder="Brief summary of this document..."
              className="flex-1 bg-background text-[11px] text-foreground/90 rounded px-2 py-0.5 border border-border/50 outline-none focus:border-primary/50 italic"
            />
          </div>
        ) : (
          fm.summary && (
            <div className="text-[11px] text-muted-foreground bg-secondary/50 px-5 py-2 flex-shrink-0 italic border-b border-border/20">
              {fm.summary}
            </div>
          )
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          {editing ? (
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className="w-full h-full resize-none bg-background text-foreground text-xs font-mono leading-relaxed p-5 outline-none"
              spellCheck={false}
            />
          ) : (
            <div className="px-5 py-4">
              <div className="prose-doc">
                <Streamdown>{doc.content}</Streamdown>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// -- Right Panel state type --
type RightPanelMode =
  | { kind: "empty" }
  | { kind: "loading-entity" }
  | { kind: "entity-detail"; detail: EntityDetailData }
  | { kind: "hover-node"; node: GNode }
  | { kind: "doc-detail"; doc: MemoryReadResult }
  | { kind: "search"; results: MemorySearchResult; query: string };

// -- Main Page --

export default function MemoryPage() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [entities, setEntities] = useState<EntitySummary[]>([]);
  const [entityTotal, setEntityTotal] = useState(0);
  const [entityConnectionCount, setEntityConnectionCount] = useState(0);
  const [searchQuery, setSearchQuery] = useState("");
  const [doc, setDoc] = useState<MemoryReadResult | null>(null);
  const [activeEntity, setActiveEntity] = useState<string | null>(null);
  const [selectedGraphNode, setSelectedGraphNode] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("entities");
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const hoverDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Right panel state
  const [rightPanel, setRightPanel] = useState<RightPanelMode>({ kind: "empty" });

  // Store graph edges data for connection counts (fetched once)
  const [graphEdges, setGraphEdges] = useState<{ nodes: EntityGraphNode[]; edges: { source: string | EntityGraphNode; target: string | EntityGraphNode; type: string }[] } | null>(null);

  // Mirror focused item for the agent. Open document wins (most specific),
  // then graph node, then active entity.
  useReportMcFocus(
    "memory",
    doc
      ? { kind: "path", id: doc.path }
      : selectedGraphNode
        ? { kind: "entity", id: selectedGraphNode }
        : activeEntity
          ? { kind: "entity", id: activeEntity }
          : null,
  );

  // Load stats and entity list on mount. The entity graph is fetched once by
  // the EntityGraph component and surfaced here via onGraphLoaded so we don't
  // pay for a second 1.1MB round-trip + parse.
  useEffect(() => {
    api.memoryStats().then(setStats).catch(console.error);
    api.entityList(500).then((d) => {
      setEntities(d.entities);
      setEntityTotal(d.total);
    }).catch(console.error);
  }, []);

  const handleGraphLoaded = useCallback((g: EntityGraphData) => {
    setEntityConnectionCount(g.edges.length);
    setGraphEdges(g);
  }, []);

  // Debounced search
  const doSearch = useCallback((q: string) => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    if (!q || q.length < 2) {
      setRightPanel((prev) => {
        // Only clear search results, don't clear entity/doc detail
        if (prev.kind === "search") return { kind: "empty" };
        return prev;
      });
      return;
    }
    searchTimerRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const results = await api.memorySearch(q, 15);
        setRightPanel({ kind: "search", results, query: q });
      } catch (err) {
        console.error("Search failed:", err);
      } finally {
        setSearching(false);
      }
    }, 300);
  }, []);

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const q = e.target.value;
    setSearchQuery(q);
    doSearch(q);
  };

  // Entity selection → set graph node + show entity detail
  const handleEntitySelect = useCallback((name: string | null) => {
    setActiveEntity(name);
    // Graph nodes are keyed by plain entity name (see /api/entity-graph)
    setSelectedGraphNode(name);

    if (!name) {
      setRightPanel({ kind: "empty" });
      return;
    }

    setRightPanel({ kind: "loading-entity" });
    api.entityDetail(name).then((d) => {
      setRightPanel({ kind: "entity-detail", detail: d });
    }).catch(console.error);
  }, []);

  // Graph node single-click handler
  const handleOpenFile = useCallback(async (path: string) => {
    try {
      const result = await api.memoryRead(path);
      setDoc(result);
    } catch (err) {
      console.error("Read failed:", err);
    }
  }, []);

  // Apply incoming focus from mc_navigate.
  // Path-shaped values (contain "/" or end in .md) open the file in the
  // explorer; bare names are treated as entities.
  const pendingFocus = usePendingFocusFor("memory");
  useEffect(() => {
    if (!pendingFocus) return;
    const looksLikePath = pendingFocus.includes("/") || pendingFocus.endsWith(".md");
    if (looksLikePath) {
      setSidebarTab("explorer");
      handleOpenFile(pendingFocus);
    } else {
      setActiveEntity(pendingFocus);
    }
  }, [pendingFocus, handleOpenFile]);

  const handleCloseDoc = useCallback(() => {
    setDoc(null);
  }, []);

  const handleGraphNodeClick = useCallback((nodeId: string | null) => {
    setSelectedGraphNode(nodeId);

    if (!nodeId) {
      setActiveEntity(null);
      setRightPanel({ kind: "empty" });
      return;
    }

    // Entity node → show entity detail
    // Support both "entity::Name" prefix and plain entity names (no "/" means not a file path)
    const isEntity = nodeId.startsWith("entity::") || !nodeId.includes("/");
    if (isEntity) {
      const entityName = nodeId.startsWith("entity::") ? nodeId.slice("entity::".length) : nodeId;
      setActiveEntity(entityName);
      setSidebarTab("entities");
      setRightPanel({ kind: "loading-entity" });
      api.entityDetail(entityName).then((d) => {
        setRightPanel({ kind: "entity-detail", detail: d });
      }).catch(console.error);
      return;
    }

    // Document node → open document modal directly
    setActiveEntity(null);
    handleOpenFile(nodeId);
  }, [handleOpenFile]);

  // Graph node double-click handler
  const handleGraphNodeDoubleClick = useCallback((nodeId: string) => {
    if (nodeId.startsWith("entity::") || !nodeId.includes("/")) {
      return;
    }
    handleOpenFile(nodeId);
  }, [handleOpenFile]);

  // Graph hover handler (debounced at EntityGraph level, but extra guard here)
  const handleGraphNodeHover = useCallback((node: GNode | null) => {
    if (hoverDebounceRef.current) clearTimeout(hoverDebounceRef.current);

    if (!node) {
      // Revert to previous sticky state (entity detail or doc detail or empty)
      hoverDebounceRef.current = setTimeout(() => {
        setRightPanel((prev) => {
          if (prev.kind === "hover-node") return { kind: "empty" };
          return prev;
        });
      }, 50);
      return;
    }

    hoverDebounceRef.current = setTimeout(() => {
      setRightPanel((prev) => {
        // Don't override sticky states with hover
        if (prev.kind === "entity-detail" || prev.kind === "doc-detail" || prev.kind === "search") {
          return prev;
        }
        return { kind: "hover-node", node };
      });
    }, 50);
  }, []);

  // Render the right panel content
  const renderRightPanel = () => {
    switch (rightPanel.kind) {
      case "search":
        return (
          <SearchResults
            results={rightPanel.results}
            query={rightPanel.query}
            onOpenFile={handleOpenFile}
          />
        );
      case "entity-detail":
        return <EntityDetailPanel detail={rightPanel.detail} />;
      case "loading-entity":
        return <div className="text-[11px] text-muted-foreground py-4">Loading facts...</div>;
      case "doc-detail":
        return <DocNodeDetail doc={rightPanel.doc} graphData={graphEdges} />;
      case "hover-node":
        return (
          <div>
            <div className="text-[9px] uppercase tracking-wider text-muted-foreground mb-2">Hover preview</div>
            <NodeQuickInfo node={rightPanel.node} graphData={graphEdges} />
          </div>
        );
      case "empty":
      default:
        return null;
    }
  };

  return (
    <div className="p-6 flex flex-col h-full min-h-0 gap-3">
      {/* Header: search bar + stats */}
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search vault (BM25 full-text)..."
            className="w-full bg-secondary text-sm text-foreground rounded-lg pl-10 pr-10 py-2 border border-border/50 outline-none focus:border-primary/50 placeholder:text-muted-foreground"
          />
          {searchQuery && (
            <button
              onClick={() => {
                setSearchQuery("");
                setRightPanel({ kind: "empty" });
                setActiveEntity(null);
              }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground/90"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {searching && (
            <div className="absolute right-10 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground">
              searching...
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-[11px] text-muted-foreground flex-shrink-0">
          {stats && (
            <span className="flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-primary" />
              <span className="text-foreground font-medium">{stats.docCount}</span> documents
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <User className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-foreground font-medium">{entityTotal}</span> entities
          </span>
          <span className="flex items-center gap-1.5">
            <Network className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-foreground font-medium">{entityConnectionCount}</span> connections
          </span>
          {stats && (
            <span className="flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-foreground font-medium">{stats.tagCount}</span> tags
            </span>
          )}
        </div>
      </div>

      {/* 3-column layout */}
      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left: Entities + Explorer */}
        <div className="w-44 flex-shrink-0 flex flex-col min-h-0">
          <div className="flex border-b border-border/50 mb-2 flex-shrink-0">
            <button
              onClick={() => setSidebarTab("entities")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                sidebarTab === "entities"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground/90"
              }`}
            >
              <User className="w-3 h-3" />
              Entities
            </button>
            <button
              onClick={() => setSidebarTab("explorer")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                sidebarTab === "explorer"
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground/90"
              }`}
            >
              <Folder className="w-3 h-3" />
              Explorer
            </button>
          </div>

          <div className="flex-1 overflow-y-auto pr-1">
            {sidebarTab === "entities" && entities.length > 0 && (
              <EntitySidebar
                entities={entities}
                activeEntity={activeEntity}
                onSelectEntity={handleEntitySelect}
              />
            )}
            {sidebarTab === "entities" && entities.length === 0 && (
              <div className="text-[11px] text-muted-foreground px-2 py-4">Loading entities...</div>
            )}
            {sidebarTab === "explorer" && (
              <VaultExplorer onOpenFile={handleOpenFile} />
            )}
          </div>
        </div>

        {/* Center: Entity/Knowledge graph (right panel floats over it) */}
        <div className="flex-1 min-w-0 min-h-0 relative">
          <EntityGraph
            selectedNode={selectedGraphNode}
            onNodeClick={handleGraphNodeClick}
            onNodeDoubleClick={handleGraphNodeDoubleClick}
            onNodeHover={handleGraphNodeHover}
            onGraphLoaded={handleGraphLoaded}
          />

          {/* Right: dynamic panel — overlays the top-right of the graph so
              the graph keeps its full width when something is selected. */}
          {rightPanel.kind !== "empty" && (
            <div className="absolute top-0 right-0 bottom-0 w-72 overflow-y-auto bg-card/85 backdrop-blur-sm border-l border-border/40 rounded-l-md shadow-lg p-3 pointer-events-auto">
              {renderRightPanel()}
            </div>
          )}
        </div>
      </div>

      {/* Document modal */}
      {doc && <DocumentModal doc={doc} onClose={handleCloseDoc} onSaved={setDoc} />}
    </div>
  );
}
