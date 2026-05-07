import { useEffect, useState } from "react";
import { Wrench, ChevronDown, ChevronRight, AlertCircle, Star, Sparkles } from "lucide-react";
import { api, type McpTool, type McpServer, type ToolsData, type ToolDiscoverySettings } from "../../api";
import { cn } from "@/lib/utils";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from "@/components/ui/collapsible";

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
          ? "bg-secondary opacity-40 cursor-not-allowed"
          : enabled
          ? "bg-primary hover:bg-primary"
          : "bg-muted hover:bg-secondary"
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
  isBaseline,
  baselineToggling,
  onBaselineToggle,
  baselineMeaningful,
}: {
  tool: McpTool;
  serverEnabled: boolean;
  toggling: boolean;
  onToggle: () => void;
  isBaseline: boolean;
  baselineToggling: boolean;
  onBaselineToggle: () => void;
  /** True when progressive discovery is enabled + active — the ★ only
   *  affects routing in that case, so we soften the affordance otherwise. */
  baselineMeaningful: boolean;
}) {
  const inactive = !serverEnabled || toggling;
  const baselineDisabled = !serverEnabled || !tool.enabled || baselineToggling;
  return (
    <div className="flex items-center gap-3 pl-10 pr-4 py-2 hover:bg-card rounded-lg group">
      <Toggle enabled={tool.enabled} onToggle={onToggle} disabled={inactive} />
      <button
        type="button"
        onClick={onBaselineToggle}
        disabled={baselineDisabled}
        title={
          isBaseline
            ? "Baseline tool — always exposed to the model"
            : "Lazy tool — loaded via ToolSearch on demand"
        }
        className={cn(
          "flex-shrink-0 -ml-1 mr-1 p-1 rounded transition-colors",
          baselineDisabled
            ? "opacity-40 cursor-not-allowed"
            : "hover:bg-accent",
          isBaseline
            ? baselineMeaningful
              ? "text-amber-400"
              : "text-amber-400/60"
            : "text-muted-foreground/50",
        )}
        aria-label={isBaseline ? "Remove from baseline" : "Add to baseline"}
      >
        <Star className={cn("w-3.5 h-3.5", isBaseline && "fill-current")} />
      </button>
      <div className="flex-1 min-w-0">
        <span
          className={cn(
            "text-sm font-mono",
            serverEnabled && tool.enabled ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {tool.name}
        </span>
        {tool.description && (
          <span className="ml-2 text-xs text-muted-foreground truncate">{tool.description}</span>
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
  baselineSet,
  onBaselineToggle,
  baselineMeaningful,
}: {
  server: McpServer;
  expanded: boolean;
  onExpandToggle: () => void;
  togglingKey: string | null;
  onServerToggle: () => void;
  onToolToggle: (toolName: string) => void;
  baselineSet: Set<string>;
  onBaselineToggle: (toolName: string) => void;
  baselineMeaningful: boolean;
}) {
  const enabledCount = server.tools.filter((t) => t.enabled).length;
  const toolCount = server.tools.length;
  const serverToggling = togglingKey === `server:${server.name}`;

  return (
    <div className="mb-4">
      {/* Server header row — bigger title, primary-accent left bar when
          enabled, description on its own line for breathing room. */}
      <div
        className={`flex items-stretch gap-3 px-4 py-3 rounded-lg cursor-pointer hover:bg-card border-l-2 ${
          server.enabled ? "border-l-primary/70" : "border-l-transparent opacity-60"
        }`}
      >
        <div className="flex items-center pt-1">
          <Toggle
            enabled={server.enabled}
            onToggle={onServerToggle}
            disabled={serverToggling}
          />
        </div>
        <button
          className="flex flex-col flex-1 min-w-0 text-left"
          onClick={onExpandToggle}
        >
          <div className="flex items-center gap-2 w-full">
            <span
              className={`text-base font-semibold tracking-tight ${
                server.enabled ? "text-foreground" : "text-muted-foreground"
              }`}
            >
              {server.label}
            </span>
            <span className="ml-auto flex items-center gap-2 flex-shrink-0">
              {server.error && (
                <span title={server.error}>
                  <AlertCircle className="w-4 h-4 text-amber-500" />
                </span>
              )}
              {toolCount > 0 && (
                <span className="text-xs font-mono text-muted-foreground tabular-nums">
                  {server.enabled ? `${enabledCount} / ${toolCount}` : toolCount}
                </span>
              )}
              {toolCount === 0 && !server.error && (
                <span className="text-xs text-muted-foreground/70 italic">no tools</span>
              )}
              <span className="text-muted-foreground/70">
                {expanded ? (
                  <ChevronDown className="w-4 h-4" />
                ) : (
                  <ChevronRight className="w-4 h-4" />
                )}
              </span>
            </span>
          </div>
          {server.description && (
            <span className="text-xs text-muted-foreground mt-0.5 leading-snug line-clamp-2">
              {server.description}
            </span>
          )}
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
          {(() => {
            const grouped = new Map<string, McpTool[]>();
            for (const t of server.tools) {
              const cat = t.category ?? "Other";
              const list = grouped.get(cat);
              if (list) list.push(t);
              else grouped.set(cat, [t]);
            }
            const categories = [...grouped.keys()].sort();
            const showHeaders = categories.length > 1;
            return categories.map((cat) => {
              const tools = grouped.get(cat)!.slice().sort((a, b) => a.name.localeCompare(b.name));
              return (
                <div key={cat} className="mt-1 first:mt-0">
                  {showHeaders && (
                    <div className="pl-10 pr-4 pt-2 pb-1 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                      {cat}
                    </div>
                  )}
                  {tools.map((tool) => (
                    <ToolRow
                      key={tool.name}
                      tool={tool}
                      serverEnabled={server.enabled}
                      toggling={togglingKey === `tool:${server.name}:${tool.name}`}
                      onToggle={() => onToolToggle(tool.name)}
                      isBaseline={baselineSet.has(tool.name)}
                      baselineToggling={togglingKey === `baseline:${tool.name}`}
                      onBaselineToggle={() => onBaselineToggle(tool.name)}
                      baselineMeaningful={baselineMeaningful}
                    />
                  ))}
                </div>
              );
            });
          })()}
          {server.tools.length === 0 && !server.error && (
            <div className="pl-10 pr-4 py-2 text-xs text-muted-foreground/70 italic">
              {server.enabled ? "No tools discovered" : "Enable server to discover tools"}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Tool Discovery (progressive disclosure) settings card ──────────────────

function ToolDiscoveryCard({
  settings,
  onChange,
}: {
  settings: ToolDiscoverySettings;
  onChange: (patch: Partial<ToolDiscoverySettings>) => void;
}) {
  const [open, setOpen] = useState(false);
  // Local draft for the number inputs so users can type freely; we only
  // commit on blur or Enter.
  const [thresholdDraft, setThresholdDraft] = useState(String(settings.threshold_tools));
  const [maxDefaultDraft, setMaxDefaultDraft] = useState(String(settings.max_results_default));
  const [maxCapDraft, setMaxCapDraft] = useState(String(settings.max_results_cap));

  // Resync drafts when settings change underneath us (e.g. from another tab).
  useEffect(() => { setThresholdDraft(String(settings.threshold_tools)) }, [settings.threshold_tools]);
  useEffect(() => { setMaxDefaultDraft(String(settings.max_results_default)) }, [settings.max_results_default]);
  useEffect(() => { setMaxCapDraft(String(settings.max_results_cap)) }, [settings.max_results_cap]);

  const commitInt = (raw: string, key: keyof ToolDiscoverySettings, current: number) => {
    const n = parseInt(raw, 10);
    if (Number.isNaN(n) || n === current) return;
    onChange({ [key]: n });
  };

  const stateChip = settings.active
    ? <span className="text-amber-400">active</span>
    : settings.enabled
    ? <span className="text-emerald-400">on (idle)</span>
    : <span className="text-muted-foreground">off</span>;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mb-4 rounded-xl border border-border bg-card overflow-hidden">
      <CollapsibleTrigger
        className="w-full flex items-center gap-3 px-4 py-3 hover:bg-accent/40 transition-colors"
      >
        <Sparkles className="w-4 h-4 text-primary flex-shrink-0" />
        <div className="flex-1 text-left min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-foreground">Tool Discovery</span>
            <span className="text-xs">{stateChip}</span>
          </div>
          <div className="text-xs text-muted-foreground mt-0.5">
            {settings.baseline_tools.length} baseline · threshold {settings.threshold_tools} ·
            {" "}{settings.total_tools || "—"} total tools · max {settings.max_results_default}/{settings.max_results_cap}
          </div>
        </div>
        {open ? <ChevronDown className="w-4 h-4 text-muted-foreground flex-shrink-0" /> : <ChevronRight className="w-4 h-4 text-muted-foreground flex-shrink-0" />}
      </CollapsibleTrigger>

      <CollapsibleContent className="px-4 pb-4 pt-1 border-t border-border/50">
        <p className="text-xs text-muted-foreground mb-4 leading-relaxed">
          When the catalog grows past <span className="font-mono">threshold_tools</span>, the
          harness advertises a small <span className="font-mono">baseline</span> + a
          <span className="font-mono"> ToolSearch</span> meta-tool to the model instead of
          every tool, and the model loads the rest on demand. Mark a tool with ★ in the
          list below to keep it in the always-on baseline.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 text-sm">
          <div className="flex items-center justify-between">
            <label htmlFor="td-enabled" className="text-foreground">Enabled</label>
            <Switch
              id="td-enabled"
              checked={settings.enabled}
              onCheckedChange={(v) => onChange({ enabled: v })}
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="td-threshold" className="text-foreground whitespace-nowrap">
              Threshold tools
            </label>
            <Input
              id="td-threshold"
              type="number"
              min={0}
              max={1000}
              value={thresholdDraft}
              onChange={(e) => setThresholdDraft(e.target.value)}
              onBlur={() => commitInt(thresholdDraft, "threshold_tools", settings.threshold_tools)}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              }}
              className="h-8 w-24 text-right"
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="td-max-default" className="text-foreground whitespace-nowrap">
              Max results (default)
            </label>
            <Input
              id="td-max-default"
              type="number"
              min={1}
              max={50}
              value={maxDefaultDraft}
              onChange={(e) => setMaxDefaultDraft(e.target.value)}
              onBlur={() => commitInt(maxDefaultDraft, "max_results_default", settings.max_results_default)}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              }}
              className="h-8 w-24 text-right"
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <label htmlFor="td-max-cap" className="text-foreground whitespace-nowrap">
              Max results (cap)
            </label>
            <Input
              id="td-max-cap"
              type="number"
              min={1}
              max={100}
              value={maxCapDraft}
              onChange={(e) => setMaxCapDraft(e.target.value)}
              onBlur={() => commitInt(maxCapDraft, "max_results_cap", settings.max_results_cap)}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
              }}
              className="h-8 w-24 text-right"
            />
          </div>
        </div>

        {settings.baseline_tools.length > 0 && (
          <div className="mt-4 pt-3 border-t border-border/50">
            <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">
              Baseline tools ({settings.baseline_tools.length})
            </div>
            <div className="flex flex-wrap gap-1.5">
              {settings.baseline_tools.map((t) => (
                <span
                  key={t}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-mono bg-amber-500/10 text-amber-300 border border-amber-500/20"
                >
                  <Star className="w-3 h-3 fill-current" /> {t}
                </span>
              ))}
            </div>
          </div>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}

// ── Main page ──────────────────────────────────────────────────────────────

export default function ToolsPage() {
  const [data, setData] = useState<ToolsData>({ servers: [] });
  const [loading, setLoading] = useState(true);
  const [togglingKey, setTogglingKey] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [discovery, setDiscovery] = useState<ToolDiscoverySettings | null>(null);

  useEffect(() => {
    Promise.all([api.tools(), api.toolDiscovery()])
      .then(([d, td]) => {
        setData(d);
        setDiscovery(td);
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

  const handleBaselineToggle = async (toolName: string) => {
    if (!discovery) return;
    const isBaseline = discovery.baseline_tools.includes(toolName);
    const newEnabled = !isBaseline;
    const key = `baseline:${toolName}`;
    setTogglingKey(key);

    // Optimistic
    setDiscovery((prev) => prev ? {
      ...prev,
      baseline_tools: newEnabled
        ? [...prev.baseline_tools, toolName]
        : prev.baseline_tools.filter((t) => t !== toolName),
    } : prev);

    try {
      await api.toolToggle({ type: "baseline", tool: toolName, enabled: newEnabled });
    } catch {
      // Revert
      setDiscovery((prev) => prev ? {
        ...prev,
        baseline_tools: isBaseline
          ? [...prev.baseline_tools, toolName]
          : prev.baseline_tools.filter((t) => t !== toolName),
      } : prev);
      setError(`Failed to ${newEnabled ? "add" : "remove"} ${toolName} ${newEnabled ? "to" : "from"} baseline`);
    } finally {
      setTogglingKey(null);
    }
  };

  const handleDiscoveryChange = async (patch: Partial<ToolDiscoverySettings>) => {
    if (!discovery) return;
    const prev = discovery;
    // Optimistic
    setDiscovery({ ...prev, ...patch });
    try {
      await api.setToolDiscovery(patch);
    } catch {
      setDiscovery(prev);
      setError("Failed to update tool discovery settings");
    }
  };

  // ── Counts + derived state ────────────────────────────────────────────

  const totalTools = data.servers.reduce((n, s) => n + s.tools.length, 0);
  const totalEnabled = data.servers.reduce(
    (n, s) => n + (s.enabled ? s.tools.filter((t) => t.enabled).length : 0),
    0,
  );

  const baselineSet = new Set(discovery?.baseline_tools ?? []);
  // The ★ only changes routing when discovery is enabled AND the catalog
  // size has crossed the threshold. Otherwise we still let the user toggle
  // (so they can pre-load a baseline before the catalog grows) but show
  // it dimmed.
  const baselineMeaningful = !!discovery?.enabled && totalTools >= (discovery?.threshold_tools ?? 30);

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Wrench className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold text-foreground">Tools</h2>
          </div>
          {!loading && totalTools > 0 && (
            <span className="text-sm text-muted-foreground">
              {totalEnabled} / {totalTools} enabled
            </span>
          )}
        </div>
        <p className="text-xs text-muted-foreground mt-1">
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
          <div className="text-sm text-muted-foreground px-4 py-8 text-center">
            Discovering tools...
          </div>
        )}

        {!loading && discovery && (
          <ToolDiscoveryCard settings={discovery} onChange={handleDiscoveryChange} />
        )}

        {!loading && (
          <>
            {/* MCP Servers section */}
            <div className="mb-2">
              <div className="flex items-center justify-between px-4 mb-2">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                  MCP Servers
                </h3>
                <div className="flex items-center gap-3">
                  <button
                    onClick={expandAll}
                    className="text-xs text-muted-foreground hover:text-foreground/90 transition-colors"
                  >
                    expand all
                  </button>
                  <span className="text-muted-foreground/50">·</span>
                  <button
                    onClick={collapseAll}
                    className="text-xs text-muted-foreground hover:text-foreground/90 transition-colors"
                  >
                    collapse all
                  </button>
                </div>
              </div>

              <div className="bg-background rounded-xl border border-border overflow-hidden px-1 py-1">
                {data.servers.map((server) => (
                  <ServerGroup
                    key={server.name}
                    server={server}
                    expanded={expanded.has(server.name)}
                    onExpandToggle={() => toggleExpand(server.name)}
                    togglingKey={togglingKey}
                    onServerToggle={() => handleServerToggle(server.name)}
                    onToolToggle={(toolName) => handleToolToggle(server.name, toolName)}
                    baselineSet={baselineSet}
                    onBaselineToggle={handleBaselineToggle}
                    baselineMeaningful={baselineMeaningful}
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
