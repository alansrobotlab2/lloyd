import { useEffect, useState, useCallback } from "react";
import { Wrench } from "lucide-react";
import { api, ToolGroupData } from "../../api";
import { getFuncGroups, getSourceMeta, getSourceKey, UNGROUPED_ICON } from "../../toolLayout";

// ── Toggle Switch ────────────────────────────────────────────────────────

function Toggle({ checked, onChange }: { checked: boolean; onChange: () => void }) {
  return (
    <button
      onClick={onChange}
      className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 cursor-pointer ${
        checked ? "bg-brand-600" : "bg-surface-3"
      }`}
    >
      <span
        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
          checked ? "left-[14px]" : "left-[2px]"
        }`}
      />
    </button>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function ToolsPage() {
  const [groups, setGroups] = useState<ToolGroupData[]>([]);

  const loadTools = useCallback(() => {
    api.tools().then((d) => setGroups(d.groups || [])).catch(console.error);
  }, []);

  useEffect(() => { loadTools(); }, [loadTools]);

  const handleToggle = async (toolName: string, currentEnabled: boolean) => {
    setGroups((prev) =>
      prev.map((g) => ({
        ...g,
        tools: g.tools.map((t) =>
          t.name === toolName ? { ...t, enabled: !currentEnabled } : t
        ),
      }))
    );
    try {
      await api.toolToggle(toolName, !currentEnabled);
    } catch (err) {
      console.error("Toggle failed:", err);
      loadTools();
    }
  };

  // Build tool status lookup
  const toolMap = new Map<string, boolean>();
  groups.forEach(g => g.tools.forEach(t => toolMap.set(t.name, t.enabled)));

  const totalTools = toolMap.size;
  const enabledTools = [...toolMap.values()].filter(Boolean).length;

  // Merge API groups by source prefix
  const mergedSources = new Map<string, Set<string>>();
  for (const g of groups) {
    const key = getSourceKey(g.source);
    if (!mergedSources.has(key)) mergedSources.set(key, new Set());
    g.tools.forEach(t => mergedSources.get(key)!.add(t.name));
  }

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Wrench className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Tools</h2>
        <span className="text-xs text-slate-500">
          {enabledTools} enabled / {totalTools} total
        </span>
      </div>

      {[...mergedSources.entries()].map(([sourceKey, sourceTools]) => {
        const meta = getSourceMeta(sourceKey);
        const SourceIcon = meta.icon;
        const funcGroups = getFuncGroups(sourceKey);

        const placed = new Set<string>();
        const renderedGroups: { label: string; icon: React.ComponentType<{ className?: string }>; color: string; tools: string[] }[] = [];

        if (funcGroups) {
          for (const fg of funcGroups) {
            const present = fg.tools.filter(t => sourceTools.has(t));
            if (present.length > 0) {
              present.forEach(t => placed.add(t));
              renderedGroups.push({ label: fg.label, icon: fg.icon, color: fg.color, tools: present });
            }
          }
        }

        const ungrouped = [...sourceTools].filter(t => !placed.has(t));
        const sourceEnabled = [...sourceTools].filter(t => toolMap.get(t)).length;

        return (
          <div key={sourceKey} className="space-y-4">
            <div className="flex items-center gap-2">
              <SourceIcon className={`w-4 h-4 ${meta.color}`} />
              <span className="text-sm font-medium text-slate-200">{meta.label}</span>
              <span className="text-[10px] text-slate-600 ml-1">{sourceEnabled}/{sourceTools.size} enabled</span>
            </div>

            {renderedGroups.map(({ label, icon: FgIcon, color, tools }) => {
              const enabledCount = tools.filter(t => toolMap.get(t)).length;
              return (
                <div key={label} className="space-y-2 ml-2">
                  <div className="flex items-center gap-2 px-1">
                    <FgIcon className={`w-3.5 h-3.5 ${color}`} />
                    <span className="text-[10px] text-slate-400 font-medium">{label}</span>
                    <span className="text-[9px] text-slate-600">{enabledCount}/{tools.length}</span>
                  </div>
                  <div className="grid grid-cols-2 xl:grid-cols-3 gap-2 ml-5">
                    {tools.map((toolName) => {
                      const enabled = toolMap.get(toolName) ?? false;
                      return (
                        <div key={toolName} className={`bg-surface-0 rounded-lg px-3 py-2 border transition-colors flex items-center gap-3 ${enabled ? "border-surface-3/50" : "border-surface-3/30 opacity-50"}`}>
                          <Toggle checked={enabled} onChange={() => handleToggle(toolName, enabled)} />
                          <span className="text-[10px] font-mono text-slate-300 truncate">{toolName}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}

            {ungrouped.length > 0 && (
              <div className="space-y-2 ml-2">
                <div className="flex items-center gap-2 px-1">
                  <UNGROUPED_ICON className="w-3.5 h-3.5 text-slate-500" />
                  <span className="text-[10px] text-slate-400 font-medium">Other</span>
                  <span className="text-[9px] text-slate-600">{ungrouped.filter(t => toolMap.get(t)).length}/{ungrouped.length}</span>
                </div>
                <div className="grid grid-cols-2 xl:grid-cols-3 gap-2 ml-5">
                  {ungrouped.map((toolName) => {
                    const enabled = toolMap.get(toolName) ?? false;
                    return (
                      <div key={toolName} className={`bg-surface-0 rounded-lg px-3 py-2 border transition-colors flex items-center gap-3 ${enabled ? "border-surface-3/50" : "border-surface-3/30 opacity-50"}`}>
                        <Toggle checked={enabled} onChange={() => handleToggle(toolName, enabled)} />
                        <span className="text-[10px] font-mono text-slate-300 truncate">{toolName}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="border-b border-surface-3/20" />
          </div>
        );
      })}
    </div>
  );
}
