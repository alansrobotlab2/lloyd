import { useEffect, useState, useCallback, useRef, useMemo } from "react";
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
} from "../../api";
import { sanitizeHtml } from "../../utils/sanitize";
import EntityGraph from "../EntityGraph";

// -- Types --

type SidebarTab = "entities" | "explorer";

const TYPE_COLORS: Record<string, string> = {
  hub: "bg-amber-400/10 text-amber-400",
  notes: "bg-slate-400/10 text-slate-400",
  "project-notes": "bg-sky-400/10 text-sky-400",
  "work-notes": "bg-indigo-400/10 text-indigo-400",
  talk: "bg-emerald-400/10 text-emerald-400",
  reference: "bg-purple-400/10 text-purple-400",
};

const CATEGORY_COLORS: Record<string, string> = {
  profile: "text-sky-400",
  preference: "text-brand-400",
  event: "text-emerald-400",
  state: "text-amber-400",
  relationship: "text-purple-400",
  people: "text-pink-400",
  project: "text-indigo-400",
};

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
            className="ml-auto text-[10px] text-brand-400 hover:text-brand-300"
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
              ? "bg-brand-600/15 text-brand-400"
              : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
          }`}
        >
          <User className="w-2.5 h-2.5 flex-shrink-0 opacity-50" />
          <span className="flex-1 text-left truncate">{name}</span>
          <span className="text-[10px] text-slate-500 flex-shrink-0">{factCount}</span>
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
      <div className="flex items-center gap-2 border-b border-surface-3/30 pb-2">
        <User className="w-3.5 h-3.5 text-brand-400 flex-shrink-0" />
        <span className="text-xs font-semibold text-slate-200">{detail.name}</span>
        <span className="ml-auto text-[10px] text-slate-500">{detail.facts.length} facts</span>
      </div>

      {categories.map((cat) => {
        const facts = byCategory.get(cat)!;
        const colorClass = CATEGORY_COLORS[cat] || "text-slate-400";
        return (
          <div key={cat}>
            <div className={`text-[10px] font-semibold uppercase tracking-wider mb-1 ${colorClass}`}>
              {cat}
            </div>
            <div className="space-y-1">
              {facts.map((f, i) => (
                <div
                  key={f.id || i}
                  className="text-[11px] text-slate-300 leading-relaxed bg-surface-2/40 rounded px-2 py-1"
                >
                  {f.fact}
                  {f.confidence < 0.8 && (
                    <span className="ml-1 text-[9px] text-slate-500 opacity-70">
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
          <div className="text-[10px] font-semibold uppercase tracking-wider mb-1 text-slate-500">
            Related docs
          </div>
          <div className="space-y-0.5">
            {detail.relationships.slice(0, 10).map((r, i) => (
              <div key={i} className="text-[10px] text-slate-400 truncate flex items-center gap-1">
                <span className={`flex-shrink-0 text-[9px] px-1 py-0.5 rounded ${
                  r.type === "tag-cluster" ? "bg-amber-400/10 text-amber-500" : "bg-slate-400/10 text-slate-500"
                }`}>
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

      const expanded = await Promise.all(
        nodes.map(async (node) => {
          if (node.type !== "dir") return node;
          if (node.name === "agents") return { ...node, loaded: false, expanded: false };
          try {
            const sub = await api.memoryBrowse(node.path);
            return {
              ...node,
              expanded: true,
              loaded: true,
              entries: sub.entries.map((e) => ({
                ...e,
                path: `${node.path}/${e.name}`,
                loaded: false,
                expanded: false,
              })),
            };
          } catch {
            return node;
          }
        })
      );
      setTree(expanded);
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
        <span className="truncate">{node.title || node.name}</span>
        {node.size != null && (
          <span className="text-[10px] text-slate-600 ml-auto pr-1">
            {node.size < 1024 ? `${node.size}B` : `${(node.size / 1024).toFixed(1)}K`}
          </span>
        )}
      </button>
    );
  };

  if (tree.length === 0) {
    return <div className="text-[11px] text-slate-500 px-2 py-4">Loading...</div>;
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
      <div className="text-center py-12 text-slate-500">
        <Search className="w-6 h-6 mx-auto mb-2 opacity-30" />
        <p className="text-sm">No results for "{query}"</p>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-[10px] text-slate-500 px-1">
        {results.results.length} results for "{query}"
      </div>
      {results.results.map((r) => (
        <button
          key={r.path}
          onClick={() => onOpenFile(r.path)}
          className="w-full text-left bg-surface-1 rounded-lg p-2.5 border border-surface-3/50 hover:border-brand-500/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <FileText className="w-3.5 h-3.5 text-brand-400 flex-shrink-0" />
            <span className="text-xs font-medium text-slate-200 truncate">
              {r.title || r.path}
            </span>
          </div>
          {(r.summary || r.snippet) && (
            <p className="text-[11px] text-slate-400 mt-1 line-clamp-2 leading-relaxed">
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

  const renderedContent = useMemo(() => {
    try {
      return sanitizeHtml(doc.content);
    } catch {
      return doc.content;
    }
  }, [doc.content]);

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
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget && !editing) onClose(); }}
    >
      <div className="bg-surface-1 border border-surface-3/50 rounded-xl shadow-2xl w-[80%] h-[80%] flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-3 border-b border-surface-3/30 flex-shrink-0">
          <div className="flex-1 min-w-0">
            <div className="text-sm font-medium text-slate-200 truncate">
              {fm.title || doc.path}
            </div>
            <div className="text-[10px] text-slate-500 font-mono">{doc.path}</div>
          </div>
          <div className="flex items-center gap-1.5">
            {editing ? (
              <>
                <button
                  onClick={handleSave}
                  disabled={saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md bg-brand-600 text-white hover:bg-brand-500 disabled:opacity-50 transition-colors"
                >
                  <Check className="w-3.5 h-3.5" />
                  {saving ? "Saving..." : "Save"}
                </button>
                <button
                  onClick={handleCancel}
                  disabled={saving}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md text-slate-400 hover:text-slate-200 hover:bg-surface-2 disabled:opacity-50 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={handleEdit}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-medium rounded-md text-slate-400 hover:text-slate-200 hover:bg-surface-2 transition-colors"
                >
                  <Pencil className="w-3.5 h-3.5" />
                  Edit
                </button>
                <button
                  onClick={onClose}
                  className="text-slate-400 hover:text-slate-200 transition-colors p-1"
                >
                  <X className="w-4 h-4" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* Frontmatter badges / editable tags */}
        <div className="flex items-center gap-2 px-5 py-2 flex-wrap flex-shrink-0 border-b border-surface-3/20">
          {fm.type && (
            <span className={`text-[10px] font-medium px-2 py-0.5 rounded ${typeClass}`}>
              {fm.type}
            </span>
          )}
          {fm.status && (
            <span className="text-[10px] text-slate-400 bg-surface-2 px-2 py-0.5 rounded">
              {fm.status}
            </span>
          )}
          {editing ? (
            <div className="flex items-center gap-1.5 flex-1 min-w-0">
              <span className="text-[10px] text-slate-500 flex-shrink-0">tags:</span>
              <input
                value={editTags}
                onChange={(e) => setEditTags(e.target.value)}
                placeholder="tag1, tag2, tag3"
                className="flex-1 bg-surface-0 text-[11px] text-slate-200 rounded px-2 py-0.5 border border-surface-3/50 outline-none focus:border-brand-500/50 font-mono"
              />
            </div>
          ) : (
            Array.isArray(fm.tags) &&
              fm.tags.map((tag: string) => (
                <span
                  key={tag}
                  className="text-[10px] text-slate-400 bg-surface-2 px-1.5 py-0.5 rounded"
                >
                  #{tag}
                </span>
              ))
          )}
          {!editing && <span className="text-[10px] text-slate-500 ml-auto">{doc.lineCount} lines</span>}
        </div>

        {/* Summary */}
        {editing ? (
          <div className="flex items-center gap-1.5 px-5 py-2 flex-shrink-0 border-b border-surface-3/20">
            <span className="text-[10px] text-slate-500 flex-shrink-0">summary:</span>
            <input
              value={editSummary}
              onChange={(e) => setEditSummary(e.target.value)}
              placeholder="Brief summary of this document..."
              className="flex-1 bg-surface-0 text-[11px] text-slate-300 rounded px-2 py-0.5 border border-surface-3/50 outline-none focus:border-brand-500/50 italic"
            />
          </div>
        ) : (
          fm.summary && (
            <div className="text-[11px] text-slate-400 bg-surface-2/50 px-5 py-2 flex-shrink-0 italic border-b border-surface-3/20">
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
              className="w-full h-full resize-none bg-surface-0 text-slate-200 text-xs font-mono leading-relaxed p-5 outline-none"
              spellCheck={false}
            />
          ) : (
            <div className="px-5 py-4">
              <div
                className="prose-doc"
                dangerouslySetInnerHTML={{ __html: renderedContent }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// -- Main Page --

export default function MemoryPage() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [entities, setEntities] = useState<EntitySummary[]>([]);
  const [entityTotal, setEntityTotal] = useState(0);
  const [entityConnectionCount, setEntityConnectionCount] = useState(0);
  const [entityDetail, setEntityDetail] = useState<EntityDetailData | null>(null);
  const [loadingEntityDetail, setLoadingEntityDetail] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MemorySearchResult | null>(null);
  const [doc, setDoc] = useState<MemoryReadResult | null>(null);
  const [activeEntity, setActiveEntity] = useState<string | null>(null);
  const [selectedGraphNode, setSelectedGraphNode] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("entities");
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showSearchResults = !!(searchResults && searchResults.results.length > 0);
  const showEntityDetail = !!entityDetail && !showSearchResults;

  // Load stats, entities, and graph connection count on mount
  useEffect(() => {
    api.memoryStats().then(setStats).catch(console.error);
    api.entityList(500).then((d) => {
      setEntities(d.entities);
      setEntityTotal(d.total);
    }).catch(console.error);
    api.entityGraph().then((g) => {
      setEntityConnectionCount(g.edges.length);
    }).catch(console.error);
  }, []);

  // Load entity detail when activeEntity changes
  useEffect(() => {
    if (!activeEntity) {
      setEntityDetail(null);
      return;
    }
    setLoadingEntityDetail(true);
    api.entityDetail(activeEntity).then((d) => {
      setEntityDetail(d);
    }).catch(console.error).finally(() => setLoadingEntityDetail(false));
  }, [activeEntity]);

  // Debounced search
  const doSearch = useCallback((q: string) => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    if (!q || q.length < 2) {
      setSearchResults(null);
      return;
    }
    searchTimerRef.current = setTimeout(async () => {
      setSearching(true);
      try {
        const results = await api.memorySearch(q, 15);
        setSearchResults(results);
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

  const handleEntitySelect = (name: string | null) => {
    setActiveEntity(name);
    setSelectedGraphNode(name);
    // Don't trigger search when selecting entities — show facts in right panel instead
    if (!name) {
      setSearchResults(null);
      setSearchQuery("");
    }
  };

  const handleGraphNodeClick = (nodeId: string | null) => {
    setSelectedGraphNode(nodeId);
    // If a node is clicked and it matches an entity, highlight it in sidebar
    if (nodeId) {
      const matchingEntity = entities.find((e) => nodeId.includes(e.name) || e.name.toLowerCase() === nodeId.toLowerCase());
      if (matchingEntity) {
        setActiveEntity(matchingEntity.name);
        setSidebarTab("entities");
      }
    } else {
      setActiveEntity(null);
    }
  };

  const handleOpenFile = async (path: string) => {
    try {
      const result = await api.memoryRead(path);
      setDoc(result);
    } catch (err) {
      console.error("Read failed:", err);
    }
  };

  const handleCloseDoc = () => {
    setDoc(null);
  };

  return (
    <div className="p-6 flex flex-col h-full min-h-0 gap-3">
      {/* Header: search bar + stats */}
      <div className="flex items-center gap-4 flex-shrink-0">
        <div className="relative flex-1">
          <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={handleSearchChange}
            placeholder="Search vault (BM25 full-text)..."
            className="w-full bg-surface-2 text-sm text-slate-200 rounded-lg pl-10 pr-10 py-2 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500"
          />
          {searchQuery && (
            <button
              onClick={() => { setSearchQuery(""); setSearchResults(null); setActiveEntity(null); }}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
            >
              <X className="w-4 h-4" />
            </button>
          )}
          {searching && (
            <div className="absolute right-10 top-1/2 -translate-y-1/2 text-[10px] text-slate-500">
              searching...
            </div>
          )}
        </div>
        <div className="flex items-center gap-4 text-[11px] text-slate-400 flex-shrink-0">
          {stats && (
            <span className="flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-brand-400" />
              <span className="text-slate-200 font-medium">{stats.docCount}</span> documents
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <User className="w-3.5 h-3.5 text-sky-400" />
            <span className="text-slate-200 font-medium">{entityTotal}</span> entities
          </span>
          <span className="flex items-center gap-1.5">
            <Network className="w-3.5 h-3.5 text-purple-400" />
            <span className="text-slate-200 font-medium">{entityConnectionCount}</span> connections
          </span>
          {stats && (
            <span className="flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-200 font-medium">{stats.tagCount}</span> tags
            </span>
          )}
        </div>
      </div>

      {/* 3-column layout */}
      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left: Entities + Explorer */}
        <div className="w-44 flex-shrink-0 flex flex-col min-h-0">
          <div className="flex border-b border-surface-3/50 mb-2 flex-shrink-0">
            <button
              onClick={() => setSidebarTab("entities")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                sidebarTab === "entities"
                  ? "border-brand-400 text-brand-400"
                  : "border-transparent text-slate-500 hover:text-slate-300"
              }`}
            >
              <User className="w-3 h-3" />
              Entities
            </button>
            <button
              onClick={() => setSidebarTab("explorer")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                sidebarTab === "explorer"
                  ? "border-brand-400 text-brand-400"
                  : "border-transparent text-slate-500 hover:text-slate-300"
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
              <div className="text-[11px] text-slate-500 px-2 py-4">Loading entities...</div>
            )}
            {sidebarTab === "explorer" && (
              <VaultExplorer onOpenFile={handleOpenFile} />
            )}
          </div>
        </div>

        {/* Center: Entity/Knowledge graph */}
        <div className="flex-1 min-w-0 min-h-0">
          <EntityGraph
            selectedNode={selectedGraphNode}
            onNodeClick={handleGraphNodeClick}
          />
        </div>

        {/* Right: Entity detail or search results */}
        <div className="w-72 flex-shrink-0 overflow-y-auto min-h-0 border-l border-surface-3/30 pl-4">
          {showSearchResults && (
            <SearchResults
              results={searchResults!}
              query={searchQuery}
              onOpenFile={handleOpenFile}
            />
          )}
          {showEntityDetail && (
            <EntityDetailPanel detail={entityDetail!} />
          )}
          {!showSearchResults && !showEntityDetail && activeEntity && loadingEntityDetail && (
            <div className="text-[11px] text-slate-500 py-4">Loading facts...</div>
          )}
        </div>
      </div>

      {/* Document modal */}
      {doc && <DocumentModal doc={doc} onClose={handleCloseDoc} onSaved={setDoc} />}
    </div>
  );
}
