import { useEffect, useState, useCallback, useRef } from "react";
import {
  Brain,
  Search,
  Network,
  BarChart3,
  FileText,
  Folder,
  ChevronRight,
  Tag,
  ArrowLeft,
  Hash,
  X,
} from "lucide-react";
import {
  api,
  type MemoryStats,
  type MemorySearchResult,
  type MemoryBrowseResult,
  type MemoryReadResult,
  type TagEntry,
} from "../../api";

// ── Types ───────────────────────────────────────────────────────────────

type ViewMode = "idle" | "search" | "browse" | "read";

const TYPE_COLORS: Record<string, string> = {
  hub: "bg-amber-400/10 text-amber-400",
  notes: "bg-slate-400/10 text-slate-400",
  "project-notes": "bg-sky-400/10 text-sky-400",
  "work-notes": "bg-indigo-400/10 text-indigo-400",
  talk: "bg-emerald-400/10 text-emerald-400",
  reference: "bg-purple-400/10 text-purple-400",
};

// ── Stats Cards ─────────────────────────────────────────────────────────

function StatsRow({ stats }: { stats: MemoryStats | null }) {
  if (!stats) return null;

  const typeEntries = Object.entries(stats.types).sort((a, b) => b[1] - a[1]);
  const maxType = typeEntries[0]?.[1] || 1;

  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-1">
          <BarChart3 className="w-4 h-4 text-indigo-400" />
          <span className="text-xs text-slate-400">Documents</span>
        </div>
        <div className="text-2xl font-semibold">{stats.docCount}</div>
        <div className="text-[10px] text-slate-500 mt-1">markdown files indexed</div>
      </div>
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-1">
          <Network className="w-4 h-4 text-emerald-400" />
          <span className="text-xs text-slate-400">Tags</span>
        </div>
        <div className="text-2xl font-semibold">{stats.tagCount}</div>
        <div className="text-[10px] text-slate-500 mt-1">unique tags across vault</div>
      </div>
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-1">
          <Brain className="w-4 h-4 text-amber-400" />
          <span className="text-xs text-slate-400">Types</span>
        </div>
        <div className="space-y-1 mt-2">
          {typeEntries.slice(0, 4).map(([type, count]) => (
            <div key={type} className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400 w-20 truncate">{type}</span>
              <div className="flex-1 h-1.5 bg-surface-2 rounded-full overflow-hidden">
                <div
                  className="h-full bg-brand-500/50 rounded-full"
                  style={{ width: `${(count / maxType) * 100}%` }}
                />
              </div>
              <span className="text-[10px] text-slate-500 w-6 text-right">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

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
  const maxCount = tags[0]?.count || 1;

  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2 mb-2 px-1">
        <Tag className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
          Tags
        </span>
        {activeTag && (
          <button
            onClick={() => onSelectTag(null)}
            className="ml-auto text-[10px] text-brand-400 hover:text-brand-300"
          >
            clear
          </button>
        )}
      </div>
      {tags.map(({ tag, count }) => (
        <button
          key={tag}
          onClick={() => onSelectTag(activeTag === tag ? null : tag)}
          className={`w-full flex items-center gap-2 px-2 py-1 rounded text-[11px] transition-colors ${
            activeTag === tag
              ? "bg-brand-600/15 text-brand-400"
              : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
          }`}
        >
          <Hash className="w-2.5 h-2.5 flex-shrink-0 opacity-50" />
          <span className="flex-1 text-left truncate">{tag}</span>
          <div className="w-12 h-1 bg-surface-2 rounded-full overflow-hidden flex-shrink-0">
            <div
              className="h-full bg-brand-500/30 rounded-full"
              style={{ width: `${(count / maxCount) * 100}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-500 w-4 text-right flex-shrink-0">{count}</span>
        </button>
      ))}
    </div>
  );
}

// ── Folder Browser ──────────────────────────────────────────────────────

function FolderBrowser({
  browse,
  currentPath,
  onNavigate,
  onOpenFile,
}: {
  browse: MemoryBrowseResult | null;
  currentPath: string;
  onNavigate: (path: string) => void;
  onOpenFile: (path: string) => void;
}) {
  const pathParts = currentPath ? currentPath.split("/") : [];

  return (
    <div>
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 mb-2 flex-wrap">
        <button
          onClick={() => onNavigate("")}
          className="text-[11px] text-brand-400 hover:text-brand-300"
        >
          vault
        </button>
        {pathParts.map((part, i) => (
          <div key={i} className="flex items-center gap-1">
            <ChevronRight className="w-3 h-3 text-slate-600" />
            <button
              onClick={() => onNavigate(pathParts.slice(0, i + 1).join("/"))}
              className="text-[11px] text-brand-400 hover:text-brand-300"
            >
              {part}
            </button>
          </div>
        ))}
      </div>

      {/* Entries */}
      {browse && (
        <div className="space-y-0.5">
          {currentPath && (
            <button
              onClick={() => {
                const parent = pathParts.slice(0, -1).join("/");
                onNavigate(parent);
              }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-slate-400 hover:bg-surface-2"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              ..
            </button>
          )}
          {browse.entries.map((entry) => (
            <button
              key={entry.name}
              onClick={() => {
                const entryPath = currentPath ? `${currentPath}/${entry.name}` : entry.name;
                if (entry.type === "dir") {
                  onNavigate(entryPath);
                } else {
                  onOpenFile(entryPath);
                }
              }}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded text-xs text-slate-300 hover:bg-surface-2 transition-colors"
            >
              {entry.type === "dir" ? (
                <Folder className="w-3.5 h-3.5 text-amber-400 flex-shrink-0" />
              ) : (
                <FileText className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
              )}
              <span className="flex-1 text-left truncate">
                {entry.type === "file" ? (entry.title || entry.name) : entry.name}
              </span>
              {entry.type === "dir" && (
                <span className="text-[10px] text-slate-500">{entry.children}</span>
              )}
              {entry.type === "file" && entry.size && (
                <span className="text-[10px] text-slate-500">
                  {entry.size < 1024 ? `${entry.size}B` : `${(entry.size / 1024).toFixed(1)}K`}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Search Results ──────────────────────────────────────────────────────

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
          className="w-full text-left bg-surface-1 rounded-lg p-3 border border-surface-3/50 hover:border-brand-500/30 transition-colors"
        >
          <div className="flex items-center gap-2">
            <FileText className="w-3.5 h-3.5 text-brand-400 flex-shrink-0" />
            <span className="text-xs font-medium text-slate-200 truncate">
              {r.title || r.path}
            </span>
            {r.score > 0 && (
              <span className="text-[10px] text-slate-500 ml-auto flex-shrink-0">
                {r.score.toFixed(2)}
              </span>
            )}
          </div>
          <div className="text-[10px] text-slate-500 font-mono mt-0.5 truncate">{r.path}</div>
          {r.snippet && (
            <p className="text-[11px] text-slate-400 mt-1.5 line-clamp-3 leading-relaxed">
              {r.snippet}
            </p>
          )}
        </button>
      ))}
    </div>
  );
}

// ── Document Viewer ─────────────────────────────────────────────────────

function DocumentViewer({
  doc,
  onClose,
}: {
  doc: MemoryReadResult;
  onClose: () => void;
}) {
  const fm = doc.frontmatter;
  const typeClass = TYPE_COLORS[fm.type] || TYPE_COLORS.notes;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2 pb-3 border-b border-surface-3/30 flex-shrink-0">
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-slate-200 truncate">
            {fm.title || doc.path}
          </div>
          <div className="text-[10px] text-slate-500 font-mono">{doc.path}</div>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-200 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Frontmatter badges */}
      <div className="flex items-center gap-2 py-2 flex-wrap flex-shrink-0">
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
        {Array.isArray(fm.tags) &&
          fm.tags.map((tag: string) => (
            <span
              key={tag}
              className="text-[10px] text-slate-400 bg-surface-2 px-1.5 py-0.5 rounded"
            >
              #{tag}
            </span>
          ))}
        <span className="text-[10px] text-slate-500 ml-auto">{doc.lineCount} lines</span>
      </div>

      {/* Summary */}
      {fm.summary && (
        <div className="text-[11px] text-slate-400 bg-surface-2/50 rounded-lg px-3 py-2 mb-2 flex-shrink-0 italic">
          {fm.summary}
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto min-h-0">
        <pre className="text-[11px] text-slate-300 leading-relaxed whitespace-pre-wrap font-mono">
          {doc.content}
        </pre>
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
  const [browse, setBrowse] = useState<MemoryBrowseResult | null>(null);
  const [browsePath, setBrowsePath] = useState("");
  const [doc, setDoc] = useState<MemoryReadResult | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>("idle");
  const [activeTag, setActiveTag] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load stats and tags on mount
  useEffect(() => {
    api.memoryStats().then(setStats).catch(console.error);
    api.memoryTags(60).then((d) => setTags(d.tags)).catch(console.error);
    api.memoryBrowse("").then((d) => { setBrowse(d); }).catch(console.error);
  }, []);

  // Debounced search
  const doSearch = useCallback((q: string) => {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    if (!q || q.length < 2) {
      setSearchResults(null);
      if (viewMode === "search") setViewMode("idle");
      return;
    }
    searchTimerRef.current = setTimeout(async () => {
      setLoading(true);
      setViewMode("search");
      try {
        const results = await api.memorySearch(q, 15);
        setSearchResults(results);
      } catch (err) {
        console.error("Search failed:", err);
      } finally {
        setLoading(false);
      }
    }, 300);
  }, [viewMode]);

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
      setViewMode("idle");
    }
  };

  const handleNavigate = async (path: string) => {
    setBrowsePath(path);
    setViewMode("browse");
    try {
      const result = await api.memoryBrowse(path);
      setBrowse(result);
    } catch (err) {
      console.error("Browse failed:", err);
    }
  };

  const handleOpenFile = async (path: string) => {
    setLoading(true);
    setViewMode("read");
    try {
      const result = await api.memoryRead(path);
      setDoc(result);
    } catch (err) {
      console.error("Read failed:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleCloseDoc = () => {
    setDoc(null);
    setViewMode(searchResults ? "search" : "browse");
  };

  return (
    <div className="p-6 flex flex-col h-full min-h-0 gap-4">
      {/* Header */}
      <div className="flex items-center gap-3 flex-shrink-0">
        <Brain className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Memory</h2>
        <span className="text-xs text-slate-500">Obsidian vault + QMD backend</span>
      </div>

      {/* Stats */}
      <div className="flex-shrink-0">
        <StatsRow stats={stats} />
      </div>

      {/* Search bar */}
      <div className="relative flex-shrink-0">
        <Search className="w-4 h-4 text-slate-500 absolute left-3 top-1/2 -translate-y-1/2" />
        <input
          type="text"
          value={searchQuery}
          onChange={handleSearchChange}
          placeholder="Search vault (BM25 full-text)..."
          className="w-full bg-surface-2 text-sm text-slate-200 rounded-lg pl-10 pr-10 py-2.5 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500"
        />
        {searchQuery && (
          <button
            onClick={() => { setSearchQuery(""); setSearchResults(null); setActiveTag(null); setViewMode("idle"); }}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
          >
            <X className="w-4 h-4" />
          </button>
        )}
        {loading && (
          <div className="absolute right-10 top-1/2 -translate-y-1/2 text-[10px] text-slate-500">
            searching...
          </div>
        )}
      </div>

      {/* Main content: sidebar + right panel */}
      <div className="flex-1 flex gap-4 min-h-0 overflow-hidden">
        {/* Left sidebar: tags + folder browser */}
        <div className="w-56 flex-shrink-0 overflow-y-auto space-y-4 pr-1">
          {/* Tags */}
          {tags.length > 0 && (
            <TagSidebar tags={tags} activeTag={activeTag} onSelectTag={handleTagSelect} />
          )}

          {/* Folder browser */}
          <div>
            <div className="flex items-center gap-2 mb-2 px-1">
              <Folder className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Browse
              </span>
            </div>
            <FolderBrowser
              browse={browse}
              currentPath={browsePath}
              onNavigate={handleNavigate}
              onOpenFile={handleOpenFile}
            />
          </div>
        </div>

        {/* Right panel */}
        <div className="flex-1 min-w-0 overflow-y-auto">
          {viewMode === "read" && doc ? (
            <DocumentViewer doc={doc} onClose={handleCloseDoc} />
          ) : viewMode === "search" && searchResults ? (
            <SearchResults
              results={searchResults}
              query={searchQuery}
              onOpenFile={handleOpenFile}
            />
          ) : (
            <div className="h-full flex items-center justify-center">
              <div className="text-center text-slate-500">
                <Search className="w-8 h-8 mx-auto mb-2 opacity-30" />
                <p className="text-sm">Search or browse your vault</p>
                <p className="text-xs mt-1">
                  {stats ? `${stats.docCount} documents, ${stats.tagCount} tags` : "Loading..."}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
