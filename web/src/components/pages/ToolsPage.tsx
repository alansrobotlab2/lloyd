import { useEffect, useState } from "react";
import { Wrench, ChevronDown, ChevronRight, AlertCircle } from "lucide-react";
import { api, type McpTool, type McpServer, type ToolsData } from "../../api";

// ── Primitives ─────────────────────────────────────────────────────────────

function Toggle({
  enabled,
  onToggle,
  disabled,
}: {
  enabled: boolean;
  onToggle: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative flex-shrink-0 w-8 h-[18px] rounded-full transition-colors ${
        disabled
          ? "bg-surface-2 opacity-40 cursor-not-allowed"
          : enabled
          ? "bg-brand-600 hover:bg-brand-500"
          : "bg-surface-3 hover:bg-surface-2"
      }`}
      aria-label={enabled ? "Disable" : "Enable"}
    >
      <span
        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
          enabled ? "left-[14px]" : "left-[2px]"
        }`}
      />
    </button>
  );
}

// ── Individual tool row (inside a server group) ────────────────────────────

function ToolRow({
  tool,
  serverEnabled,
  toggling,
  onToggle,
}: {
  tool: McpTool;
  serverEnabled: boolean;
  toggling: boolean;
  onToggle: () => void;
}) {
  const inactive = !serverEnabled || toggling;
  return (
    <div className="flex items-center gap-3 pl-10 pr-4 py-2 hover:bg-surface-1 rounded-lg group">
      <Toggle enabled={tool.enabled} onToggle={onToggle} disabled={inactive} />
      <div className="flex-1 min-w-0">
        <span
          className={`text-sm font-mono ${
            serverEnabled && tool.enabled ? "text-slate-200" : "text-slate-500"
          }`}
        >
          {tool.name}
        </span>
        {tool.description && (
          <span className="ml-2 text-xs text-slate-500 truncate">{tool.description}</span>
        )}
      </div>
    </div>
  );
}

// ── Server group (collapsible) ─────────────────────────────────────────────

function ServerGroup({
  server,
  expanded,
  onExpandToggle,
  togglingKey,
  onServerToggle,
  onToolToggle,
}: {
  server: McpServer;
  expanded: boolean;
  onExpandToggle: () => void;
  togglingKey: string | null;
  onServerToggle: () => void;
  onToolToggle: (toolName: string) => void;
}) {
  const enabledCount = server.tools.filter((t) => t.enabled).length;
  const toolCount = server.tools.length;
  const serverToggling = togglingKey === `server:${server.name}`;

  return (
    <div className="mb-3">
      {/* Server header row */}
      <div
        className={`flex items-center gap-2 px-4 py-2.5 rounded-lg cursor-pointer hover:bg-surface-1 ${
          server.enabled ? "" : "opacity-60"
        }`}
      >
        <Toggle
          enabled={server.enabled}
          onToggle={onServerToggle}
          disabled={serverToggling}
        />
        <button
          className="flex items-center gap-2 flex-1 min-w-0 text-left"
          onClick={onExpandToggle}
        >
          <span
            className={`text-sm font-semibold ${
              server.enabled ? "text-slate-200" : "text-slate-500"
            }`}
          >
            {server.label}
          </span>
          {server.description && (
            <span className="text-xs text-slate-500 hidden sm:inline">{server.description}</span>
          )}
          <span className="ml-auto flex items-center gap-2 flex-shrink-0">
            {server.error && (
              <span title={server.error}>
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
              </span>
            )}
            {toolCount > 0 && (
              <span className="text-xs text-slate-500">
                {server.enabled ? `${enabledCount} / ${toolCount}` : toolCount}
              </span>
            )}
            {toolCount === 0 && !server.error && (
              <span className="text-xs text-slate-600 italic">no tools</span>
            )}
            <span className="text-slate-600">
              {expanded ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </span>
          </span>
        </button>
      </div>

      {/* Expanded tool list */}
      {expanded && (
        <div className="mt-0.5">
          {server.error && server.tools.length === 0 && (
            <div className="pl-10 pr-4 py-2 text-xs text-amber-500/80 italic">
              {server.error}
            </div>
          )}
          {server.tools.map((tool) => (
            <ToolRow
              key={tool.name}
              tool={tool}
              serverEnabled={server.enabled}
              toggling={togglingKey === `tool:${server.name}:${tool.name}`}
              onToggle={() => onToolToggle(tool.name)}
            />
          ))}
          {server.tools.length === 0 && !server.error && (
            <div className="pl-10 pr-4 py-2 text-xs text-slate-600 italic">
              {server.enabled ? "No tools discovered" : "Enable server to discover tools"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ToolsPage() {
  const [data, setData] = useState<ToolsData>({ servers: [] });
  const [loading, setLoading] = useState(true);
  const [togglingKey, setTogglingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    api
      .tools()
      .then((d) => {
        setData(d);
        // Expand all servers that have tools by default
        setExpanded(new Set(d.servers.filter((s) => s.tools.length > 0).map((s) => s.name)));
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const toggleExpand = (name: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const expandAll = () => setExpanded(new Set(data.servers.map((s) => s.name)));
  const collapseAll = () => setExpanded(new Set());

  // ── Optimistic toggle helpers ──────────────────────────────────────────

  const handleServerToggle = async (serverName: string) => {
    const server = data.servers.find((s) => s.name === serverName);
    if (!server) return;
    const newEnabled = !server.enabled;
    const key = `server:${serverName}`;
    setTogglingKey(key);

    // Optimistic
    setData((prev) => ({
      ...prev,
      servers: prev.servers.map((s) =>
        s.name === serverName ? { ...s, enabled: newEnabled } : s
      ),
    }));

    try {
      await api.toolToggle({ type: "server", server: serverName, enabled: newEnabled });
    } catch {
      // Revert
      setData((prev) => ({
        ...prev,
        servers: prev.servers.map((s) =>
          s.name === serverName ? { ...s, enabled: server.enabled } : s
        ),
      }));
      setError(`Failed to toggle ${serverName}`);
    } finally {
      setTogglingKey(null);
    }
  };

  const handleToolToggle = async (serverName: string, toolName: string) => {
    const server = data.servers.find((s) => s.name === serverName);
    const tool = server?.tools.find((t) => t.name === toolName);
    if (!tool) return;
    const newEnabled = !tool.enabled;
    const key = `tool:${serverName}:${toolName}`;
    setTogglingKey(key);

    // Optimistic
    setData((prev) => ({
      ...prev,
      servers: prev.servers.map((s) =>
        s.name === serverName
          ? {
              ...s,
              tools: s.tools.map((t) =>
                t.name === toolName ? { ...t, enabled: newEnabled } : t
              ),
            }
          : s
      ),
    }));

    try {
      await api.toolToggle({ type: "tool", server: serverName, tool: toolName, enabled: newEnabled });
    } catch {
      setData((prev) => ({
        ...prev,
        servers: prev.servers.map((s) =>
          s.name === serverName
            ? {
                ...s,
                tools: s.tools.map((t) =>
                  t.name === toolName ? { ...t, enabled: tool.enabled } : t
                ),
              }
            : s
        ),
      }));
      setError(`Failed to toggle ${toolName}`);
    } finally {
      setTogglingKey(null);
    }
  };

  // ── Counts ────────────────────────────────────────────────────────────

  const totalTools = data.servers.reduce((n, s) => n + s.tools.length, 0);
  const totalEnabled = data.servers.reduce(
    (n, s) => n + (s.enabled ? s.tools.filter((t) => t.enabled).length : 0),
    0,
  );

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Wrench className="w-5 h-5 text-brand-400" />
            <h2 className="text-lg font-semibold text-slate-200">Tools</h2>
          </div>
          {!loading && totalTools > 0 && (
            <span className="text-sm text-slate-400">
              {totalEnabled} / {totalTools} enabled
            </span>
          )}
        </div>
        <p className="text-xs text-slate-500 mt-1">
          Enable or disable MCP servers and individual tools. Changes take effect on the next session.
        </p>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        {error && (
          <div className="mb-4 mx-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400 flex items-center justify-between">
            <span>{error}</span>
            <button onClick={() => setError(null)} className="ml-3 text-red-400/60 hover:text-red-400">✕</button>
          </div>
        )}

        {loading && (
          <div className="text-sm text-slate-500 px-4 py-8 text-center">
            Discovering tools...
          </div>
        )}

        {!loading && (
          <>
            {/* MCP Servers section */}
            <div className="mb-2">
              <div className="flex items-center justify-between px-4 mb-2">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                  MCP Servers
                </h3>
                <div className="flex items-center gap-3">
                  <button
                    onClick={expandAll}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    expand all
                  </button>
                  <span className="text-slate-700">·</span>
                  <button
                    onClick={collapseAll}
                    className="text-xs text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    collapse all
                  </button>
                </div>
              </div>

              <div className="bg-surface-0 rounded-xl border border-slate-700 overflow-hidden px-1 py-1">
                {data.servers.map((server) => (
                  <ServerGroup
                    key={server.name}
                    server={server}
                    expanded={expanded.has(server.name)}
                    onExpandToggle={() => toggleExpand(server.name)}
                    togglingKey={togglingKey}
                    onServerToggle={() => handleServerToggle(server.name)}
                    onToolToggle={(toolName) => handleToolToggle(server.name, toolName)}
                  />
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
