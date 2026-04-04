import { useEffect, useState } from "react";
import { Wrench } from "lucide-react";
import { api, type ToolEntry, type ToolsData } from "../../api";

function ToggleSwitch({ enabled, onToggle, disabled }: { enabled: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
        disabled ? "bg-surface-2 opacity-40 cursor-not-allowed" : enabled ? "bg-brand-600" : "bg-surface-3"
      }`}
      aria-label={enabled ? "Disable tool" : "Enable tool"}
    >
      <span
        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
          enabled ? "left-[14px]" : "left-[2px]"
        }`}
      />
    </button>
  );
}

function ToolRow({ tool, onToggle, toggling }: { tool: ToolEntry; onToggle: (tool: ToolEntry) => void; toggling: boolean }) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5 hover:bg-surface-1 rounded-lg group">
      <ToggleSwitch enabled={tool.enabled} onToggle={() => onToggle(tool)} disabled={toggling} />
      <div className="flex-1 min-w-0">
        <span className="text-sm font-medium text-slate-200">{tool.label}</span>
        {tool.description && (
          <span className="ml-2 text-xs text-slate-400">{tool.description}</span>
        )}
      </div>
      <span className="text-xs font-mono text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
        {tool.name}
      </span>
    </div>
  );
}

function ToolGroup({ title, tools, onToggle, togglingName }: {
  title: string;
  tools: ToolEntry[];
  onToggle: (tool: ToolEntry) => void;
  togglingName: string | null;
}) {
  if (tools.length === 0) return null;
  const enabledCount = tools.filter(t => t.enabled).length;
  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-2 px-1">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{title}</h3>
        <span className="text-xs text-slate-400">{enabledCount} / {tools.length} enabled</span>
      </div>
      <div className="bg-surface-0 rounded-xl border border-slate-700 overflow-hidden">
        {tools.map((tool) => (
          <ToolRow
            key={tool.name}
            tool={tool}
            onToggle={onToggle}
            toggling={togglingName === tool.name}
          />
        ))}
      </div>
    </div>
  );
}

export default function ToolsPage() {
  const [data, setData] = useState<ToolsData>({ builtin: [], plugins: [] });
  const [togglingName, setTogglingName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.tools()
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  const handleToggle = async (tool: ToolEntry) => {
    const newEnabled = !tool.enabled;
    setTogglingName(tool.name);

    // Optimistic update
    const patch = (list: ToolEntry[]) =>
      list.map(t => t.name === tool.name ? { ...t, enabled: newEnabled } : t);
    setData(prev => ({ builtin: patch(prev.builtin), plugins: patch(prev.plugins) }));

    try {
      await api.toolToggle(tool.name, newEnabled);
    } catch (e) {
      // Revert on failure
      const revert = (list: ToolEntry[]) =>
        list.map(t => t.name === tool.name ? { ...t, enabled: tool.enabled } : t);
      setData(prev => ({ builtin: revert(prev.builtin), plugins: revert(prev.plugins) }));
      setError(`Failed to toggle ${tool.name}`);
    } finally {
      setTogglingName(null);
    }
  };

  const totalEnabled = data.builtin.filter(t => t.enabled).length + data.plugins.filter(t => t.enabled).length;
  const total = data.builtin.length + data.plugins.length;

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="flex-shrink-0 px-6 pt-5 pb-4 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Wrench className="w-5 h-5 text-brand-400" />
            <h2 className="text-lg font-semibold text-slate-200">Tools</h2>
          </div>
          {total > 0 && (
            <span className="text-sm text-slate-400">{totalEnabled} / {total} enabled</span>
          )}
        </div>
        <p className="text-xs text-slate-400 mt-1">
          Enable or disable toolsets for the CLI platform. Changes persist to config.yaml.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {error && (
          <div className="mb-4 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-sm text-red-400">
            {error}
          </div>
        )}

        {total === 0 && !error && (
          <div className="text-sm text-slate-400">Loading...</div>
        )}

        <ToolGroup
          title="Built-in"
          tools={data.builtin}
          onToggle={handleToggle}
          togglingName={togglingName}
        />
        <ToolGroup
          title="Plugins"
          tools={data.plugins}
          onToggle={handleToggle}
          togglingName={togglingName}
        />
      </div>
    </div>
  );
}
