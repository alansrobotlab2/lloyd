import { useEffect, useState, useCallback } from "react";
import { Cpu, Server, Zap, Globe, Shield, Image, Type, DollarSign } from "lucide-react";
import { api } from "../../api";

interface ModelInfo {
  id: string;
  name: string;
  contextWindow: number;
  maxTokens: number;
  reasoning: boolean;
  enabled: boolean;
  input: string[];
  cost: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number };
}

interface ProviderInfo {
  baseUrl: string;
  api: string;
  models: ModelInfo[];
}

const PROVIDER_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  "local-llm": Server,
  openrouter: Globe,
  anthropic: Shield,
};

export default function ModelsPage() {
  const [providers, setProviders] = useState<Record<string, ProviderInfo>>({});

  const loadModels = useCallback(() => {
    api.models().then((d: any) => setProviders(d.providers || {})).catch(console.error);
  }, []);

  useEffect(() => {
    loadModels();
  }, [loadModels]);

  const handleToggle = async (providerName: string, modelId: string, currentEnabled: boolean) => {
    // Optimistic update
    setProviders((prev) => {
      const updated = { ...prev };
      const provider = { ...updated[providerName] };
      provider.models = provider.models.map((m) =>
        m.id === modelId ? { ...m, enabled: !currentEnabled } : m,
      );
      updated[providerName] = provider;
      return updated;
    });

    try {
      await api.modelToggle(providerName, modelId, !currentEnabled);
    } catch (err) {
      console.error("Toggle failed:", err);
      loadModels(); // revert on error
    }
  };

  const totalModels = Object.values(providers).reduce((sum, p) => sum + p.models.length, 0);
  const enabledModels = Object.values(providers).reduce(
    (sum, p) => sum + p.models.filter((m) => m.enabled).length,
    0,
  );

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Cpu className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Models</h2>
        <span className="text-xs text-slate-500">
          {enabledModels} enabled / {totalModels} total across{" "}
          {Object.keys(providers).length} providers
        </span>
      </div>

      {Object.entries(providers).map(([name, provider]) => {
        const Icon = PROVIDER_ICONS[name] || Cpu;
        return (
          <div key={name} className="space-y-3">
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 text-slate-400" />
              <h3 className="text-sm font-medium text-slate-300">{name}</h3>
              <span className="text-[10px] text-slate-500 font-mono">{provider.baseUrl}</span>
              <span className="ml-auto text-[10px] text-slate-500 px-2 py-0.5 rounded bg-surface-2">
                {provider.api}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {provider.models.map((model) => (
                <div
                  key={model.id}
                  className={`bg-surface-1 rounded-xl p-4 border transition-colors ${
                    model.enabled
                      ? "border-surface-3/50"
                      : "border-surface-3/30 opacity-50"
                  }`}
                >
                  <div className="flex items-center gap-4">
                    {/* Toggle */}
                    <button
                      onClick={() => handleToggle(name, model.id, model.enabled)}
                      className={`relative w-9 h-5 rounded-full transition-colors flex-shrink-0 ${
                        model.enabled ? "bg-brand-600" : "bg-surface-3"
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                          model.enabled ? "left-[18px]" : "left-0.5"
                        }`}
                      />
                    </button>

                    {/* Name + ID */}
                    <div className="flex-1 min-w-0">
                      <div className="text-sm font-medium text-slate-200 truncate">
                        {model.name}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5">{model.id}</div>
                    </div>

                    {/* Badges */}
                    <div className="flex items-center gap-2 flex-shrink-0">
                      {/* Input modalities */}
                      {model.input?.includes("image") && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-purple-400/10 text-purple-400" title="Vision">
                          <Image className="w-2.5 h-2.5 inline mr-0.5" />
                          vision
                        </span>
                      )}
                      {model.reasoning && (
                        <span className="px-1.5 py-0.5 rounded text-[10px] bg-amber-400/10 text-amber-400 font-medium">
                          <Zap className="w-2.5 h-2.5 inline mr-0.5" />
                          reasoning
                        </span>
                      )}
                    </div>

                    {/* Stats */}
                    <div className="flex items-center gap-4 text-xs text-slate-400 flex-shrink-0">
                      <div className="text-right">
                        <div className="text-slate-300 font-mono">
                          {(model.contextWindow / 1000).toFixed(0)}K
                        </div>
                        <div className="text-[10px] text-slate-500">context</div>
                      </div>
                      <div className="text-right">
                        <div className="text-slate-300 font-mono">
                          {(model.maxTokens / 1000).toFixed(1)}K
                        </div>
                        <div className="text-[10px] text-slate-500">max output</div>
                      </div>
                      {model.cost?.input != null && model.cost.input > 0 && (
                        <div className="text-right">
                          <div className="text-slate-300 font-mono text-[10px]">
                            ${model.cost.input}/{model.cost.output}
                          </div>
                          <div className="text-[10px] text-slate-500">in/out per 1M</div>
                        </div>
                      )}
                      {model.cost?.input === 0 && (
                        <div className="text-right">
                          <div className="text-emerald-400 font-mono text-[10px]">free</div>
                          <div className="text-[10px] text-slate-500">cost</div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
