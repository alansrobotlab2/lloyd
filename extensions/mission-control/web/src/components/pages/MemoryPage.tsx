import { useEffect, useState, useCallback, useRef, useMemo } from "react";
import {
  Search,
  FileText,
  Folder,
  FolderOpen,
  ChevronRight,
  ChevronDown,
  Tag,
  Hash,
  X,
  Pencil,
  Check,
} from "lucide-react";
import { marked } from "marked";
import {
  api,
  type MemoryStats,
  type MemorySearchResult,
  type MemoryReadResult,
  type TagEntry,
} from "../../api";
import TagGraph from "../TagGraph";

// ── Types ───────────────────────────────────────────────────────────────

type SidebarTab = "tags" | "explorer";

const TYPE_COLORS: Record<string, string> = {
  hub: "bg-amber-400/10 text-amber-400",
  notes: "bg-slate-400/10 text-slate-400",
  "project-notes": "bg-sky-400/10 text-sky-400",
  "work-notes": "bg-indigo-400/10 text-indigo-400",
  talk: "bg-emerald-400/10 text-emerald-400",
  reference: "bg-purple-400/10 text-purple-400",
};

// ── Markdown renderer config ─────────────────────────────────────────────

marked.setOptions({
  breaks: true,
  gfm: true,
});

// ── Tag Sidebar ─────────────────────────────────────────────────────────

function TagSidebar({
  tags,
  activeTag,
  onSelectTag,
}: {
  tags: TagEntry[];
  activeTag: string | null;
  onSelectTag: (tag: string | null) => void;
}) {
  const activeRef = useRef<HTMLButtonElement>(null);

  // Scroll active tag into view when it changes (e.g. from graph click)
  useEffect(() => {
    if (activeTag && activeRef.current) {
      activeRef.current.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [activeTag]);

  return (
    <div className="space-y-1">
      {activeTag && (
        <div className="flex items-center px-1 mb-1">
          <button
            onClick={() => onSelectTag(null)}
            className="ml-auto text-[10px] text-brand-400 hover:text-brand-300"
          >
            clear
          </button>
        </div>
      )}
      {tags.map(({ tag, count }) => (
        <button
          key={tag}
          ref={activeTag === tag ? activeRef : undefined}
          onClick={() => onSelectTag(activeTag === tag ? null : tag)}
          className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] transition-colors ${
            activeTag === tag
              ? "bg-brand-600/15 text-brand-400"
              : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
          }`}
        >
          <Hash className="w-2.5 h-2.5 flex-shrink-0 opacity-50" />
          <span className="flex-1 text-left truncate">{tag}</span>
          <span className="text-[10px] text-slate-500 flex-shrink-0">{count}</span>
        </button>
      ))}
    </div>
  );
}

// ── Vault Explorer (tree view) ──────────────────────────────────────────

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

  // Load root + auto-expand first-level dirs
  useEffect(() => {
    api.memoryBrowse("").then(async (root) => {
      const nodes: TreeNode[] = root.entries.map((e) => ({
        ...e,
        path: e.name,
        loaded: false,
        expanded: false,
      }));

      // Auto-expand each top-level directory (except "agents")
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

// ── Search Results (right panel listing) ────────────────────────────────

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

// ── Document Modal ──────────────────────────────────────────────────────

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
      return marked.parse(doc.content) as string;
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

// ── Main Page ───────────────────────────────────────────────────────────

export default function MemoryPage() {
  const [stats, setStats] = useState<MemoryStats | null>(null);
  const [tags, setTags] = useState<TagEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<MemorySearchResult | null>(null);
  const [doc, setDoc] = useState<MemoryReadResult | null>(null);
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [searching, setSearching] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>("tags");
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const showRightPanel = !!(searchResults && searchResults.results.length > 0);

  // Load stats and tags on mount
  useEffect(() => {
    api.memoryStats().then(setStats).catch(console.error);
    api.memoryTags(500).then((d) => setTags(d.tags)).catch(console.error);
  }, []);

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

  const handleTagSelect = (tag: string | null) => {
    setActiveTag(tag);
    if (tag) {
      setSearchQuery(tag);
      doSearch(tag);
    } else {
      setSearchQuery("");
      setSearchResults(null);
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
      {/* Header: search bar (left half) + stats (right half) */}
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
              onClick={() => { setSearchQuery(""); setSearchResults(null); setActiveTag(null); }}
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
        {stats && (
          <div className="flex items-center gap-4 text-[11px] text-slate-400 flex-shrink-0">
            <span className="flex items-center gap-1.5">
              <FileText className="w-3.5 h-3.5 text-brand-400" />
              <span className="text-slate-200 font-medium">{stats.docCount}</span> documents
            </span>
            <span className="flex items-center gap-1.5">
              <Tag className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-slate-200 font-medium">{stats.tagCount}</span> tags
            </span>
          </div>
        )}
      </div>

      {/* 3-column layout: left sidebar | center | right results */}
      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left: Tags + Explorer */}
        <div className="w-44 flex-shrink-0 flex flex-col min-h-0">
          {/* Tab bar */}
          <div className="flex border-b border-surface-3/50 mb-2 flex-shrink-0">
            <button
              onClick={() => setSidebarTab("tags")}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider border-b-2 transition-colors ${
                sidebarTab === "tags"
                  ? "border-brand-400 text-brand-400"
                  : "border-transparent text-slate-500 hover:text-slate-300"
              }`}
            >
              <Tag className="w-3 h-3" />
              Tags
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

          {/* Tab content */}
          <div className="flex-1 overflow-y-auto pr-1">
            {sidebarTab === "tags" && tags.length > 0 && (
              <TagSidebar tags={tags} activeTag={activeTag} onSelectTag={handleTagSelect} />
            )}
            {sidebarTab === "explorer" && (
              <VaultExplorer onOpenFile={handleOpenFile} />
            )}
          </div>
        </div>

        {/* Center: Tag graph */}
        <div className="flex-1 min-w-0 min-h-0">
          <TagGraph
            selectedTag={activeTag}
            onTagClick={(tag) => {
              if (tag) {
                setSearchQuery(tag);
                setActiveTag(tag);
                setSidebarTab("tags");
                doSearch(tag);
              } else {
                // Only deselect the active tag — don't clear search results.
                // The user may be clicking a search result while a graph node
                // background-click fires simultaneously.
                setActiveTag(null);
              }
            }}
          />
        </div>

        {/* Right: Document listing (always mounted to keep graph bounds stable) */}
        <div className="w-72 flex-shrink-0 overflow-y-auto min-h-0 border-l border-surface-3/30 pl-4">
          {showRightPanel && (
            <SearchResults
              results={searchResults!}
              query={searchQuery}
              onOpenFile={handleOpenFile}
            />
          )}
        </div>
      </div>

      {/* Document modal */}
      {doc && <DocumentModal doc={doc} onClose={handleCloseDoc} onSaved={setDoc} />}
    </div>
  );
}
