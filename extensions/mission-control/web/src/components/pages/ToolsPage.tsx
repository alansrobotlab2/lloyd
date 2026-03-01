import { useEffect, useState, useCallback } from "react";
import { Wrench, Terminal, Mic, LayoutGrid, Box, GitFork, Clock } from "lucide-react";
import { api, ToolGroupData } from "../../api";

const SOURCE_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  "openclaw — sessions & agents": GitFork,
  "openclaw — files & runtime": Box,
  "openclaw — web & memory": Box,
  "openclaw — system & media": Clock,
  "mcp-tools": Terminal,
  "clawdeck": LayoutGrid,
  "voice-tools": Mic,
};

const SOURCE_COLORS: Record<string, string> = {
  "openclaw — sessions & agents": "text-cyan-400",
  "openclaw — files & runtime": "text-sky-400",
  "openclaw — web & memory": "text-violet-400",
  "openclaw — system & media": "text-orange-400",
  "mcp-tools": "text-indigo-400",
  "clawdeck": "text-emerald-400",
  "voice-tools": "text-amber-400",
};

export default function ToolsPage() {
  const [groups, setGroups] = useState<ToolGroupData[]>([]);

  const loadTools = useCallback(() => {
    api.tools().then((d) => setGroups(d.groups || [])).catch(console.error);
  }, []);

  useEffect(() => {
    loadTools();
  }, [loadTools]);

  const handleToggle = async (toolName: string, currentEnabled: boolean) => {
    // Optimistic update
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        tools: g.tools.map((t) =>
          t.name === toolName ? { ...t, enabled: !currentEnabled } : t,
        ),
      })),
    );

    try {
      await api.toolToggle(toolName, !currentEnabled);
    } catch (err) {
      console.error("Toggle failed:", err);
      loadTools(); // revert on error
    }
  };

  const totalTools = groups.reduce((sum, g) => sum + g.tools.length, 0);
  const enabledTools = groups.reduce(
    (sum, g) => sum + g.tools.filter((t) => t.enabled).length,
    0,
  );
  const builtInGroups = groups.filter((g) => g.source.startsWith("openclaw"));
  const extensionGroups = groups.filter((g) => !g.source.startsWith("openclaw"));

  const renderGroup = (group: ToolGroupData) => {
    const Icon = SOURCE_ICONS[group.source] || Wrench;
    const color = SOURCE_COLORS[group.source] || "text-slate-400";
    const enabledCount = group.tools.filter((t) => t.enabled).length;

    return (
      <div key={group.source} className="space-y-3">
        <div className="flex items-center gap-2">
          <Icon className={`w-4 h-4 ${color}`} />
          <h3 className="text-sm font-medium text-slate-300">{group.source}</h3>
          <span className="text-[10px] text-slate-500">
            {enabledCount}/{group.tools.length} enabled
          </span>
        </div>
        <div className="grid grid-cols-2 xl:grid-cols-3 gap-2">
          {group.tools.map((tool) => (
            <div
              key={tool.name}
              className={`bg-surface-1 rounded-lg px-3 py-2.5 border transition-colors flex items-center gap-3 ${
                tool.enabled
                  ? "border-surface-3/50"
                  : "border-surface-3/30 opacity-50"
              }`}
            >
              <button
                onClick={() => handleToggle(tool.name, tool.enabled)}
                className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
                  tool.enabled ? "bg-brand-600" : "bg-surface-3"
                }`}
              >
                <span
                  className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
                    tool.enabled ? "left-[14px]" : "left-[2px]"
                  }`}
                />
              </button>
              <div className="text-xs font-mono text-slate-300 truncate">
                {tool.name}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Wrench className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Tools</h2>
        <span className="text-xs text-slate-500">
          {enabledTools} enabled / {totalTools} total across {groups.length} providers
        </span>
      </div>

      {/* Extensions section */}
      {extensionGroups.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Extensions
          </div>
          <div className="space-y-4">
            {extensionGroups.map(renderGroup)}
          </div>
        </div>
      )}

      {/* Built-in section */}
      {builtInGroups.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Built-in
          </div>
          <div className="space-y-4">
            {builtInGroups.map(renderGroup)}
          </div>
        </div>
      )}
    </div>
  );
}
