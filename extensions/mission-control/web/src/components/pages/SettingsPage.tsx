import { useState } from "react";
import { Settings, Shield, Bot, Palette, Lock } from "lucide-react";

type Tab = "configuration" | "integrations" | "agents" | "interface";

const TABS: { id: Tab; label: string }[] = [
  { id: "configuration", label: "Configuration" },
  { id: "integrations", label: "Integrations" },
  { id: "agents", label: "Agents" },
  { id: "interface", label: "Interface" },
];

function ConfigurationTab() {
  return (
    <div className="space-y-4">
      <div className="bg-surface-2 rounded-lg p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-3">
          <Shield className="w-4 h-4 text-indigo-400" />
          <span className="text-sm font-medium text-slate-300">Gateway</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-slate-500 mb-1">Mode</div>
            <div className="text-slate-300 font-mono">local</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Auth</div>
            <div className="text-slate-300 font-mono">none</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Port</div>
            <div className="text-slate-300 font-mono">18789</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Hooks</div>
            <div className="text-slate-300 font-mono">enabled</div>
          </div>
        </div>
      </div>

      <div className="bg-surface-2 rounded-lg p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-3">
          <Bot className="w-4 h-4 text-emerald-400" />
          <span className="text-sm font-medium text-slate-300">Memory</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-slate-500 mb-1">Backend</div>
            <div className="text-slate-300 font-mono">qmd</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Search Mode</div>
            <div className="text-slate-300 font-mono">hybrid (0.7v / 0.3t)</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Session Retention</div>
            <div className="text-slate-300 font-mono">30 days</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Compaction</div>
            <div className="text-slate-300 font-mono">safeguard</div>
          </div>
        </div>
      </div>

      <div className="bg-surface-2 rounded-lg p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-3">
          <Lock className="w-4 h-4 text-amber-400" />
          <span className="text-sm font-medium text-slate-300">TLS Certificate</span>
        </div>
        <a
          href="/api/mc/cert"
          download
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded bg-brand-600 hover:bg-brand-500 text-white transition-colors"
        >
          Download Certificate
        </a>
        <div className="mt-3 space-y-2 text-xs text-slate-400">
          <div>
            <p className="font-medium text-slate-300">Access URLs:</p>
            <ul className="mt-1 space-y-0.5">
              <li>https://192.168.50.108:18790/</li>
              <li>https://alansrobotlab.asuscomm.com:18790/</li>
            </ul>
          </div>
          <div>
            <p className="font-medium text-slate-300">Chrome Certificate Install:</p>
            <ol className="mt-1 list-decimal list-inside space-y-1">
              <li>Download the certificate using the button above</li>
              <li>Open chrome://settings/certificates</li>
              <li>Go to the Authorities tab</li>
              <li>Click Import, select the downloaded .crt file</li>
              <li>Check "Trust this certificate for identifying websites"</li>
              <li>Click OK and restart the browser</li>
            </ol>
          </div>
        </div>
      </div>
    </div>
  );
}

function IntegrationsTab() {
  return (
    <div className="space-y-4">
      {["mcp-tools", "voice-tools", "model-router", "timing-profiler"].map(
        (plugin) => (
          <div
            key={plugin}
            className="bg-surface-2 rounded-lg p-4 border border-surface-3/50 flex items-center gap-3"
          >
            <div className="w-2 h-2 rounded-full bg-emerald-400" />
            <span className="text-sm text-slate-300 flex-1">{plugin}</span>
            <span className="text-[10px] text-slate-500 font-mono">loaded</span>
          </div>
        ),
      )}
    </div>
  );
}

function AgentsTab() {
  return (
    <div className="space-y-4">
      <div className="bg-surface-2 rounded-lg p-4 border border-surface-3/50">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-brand-600 flex items-center justify-center text-lg">
            L
          </div>
          <div>
            <div className="text-sm font-medium text-slate-200">Lloyd</div>
            <div className="text-[10px] text-slate-500 font-mono">agent:main</div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-slate-500 mb-1">Primary Model</div>
            <div className="text-slate-300 font-mono">anthropic/claude-sonnet-4-6</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Max Concurrent</div>
            <div className="text-slate-300 font-mono">4 (subagents: 8)</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function InterfaceTab() {
  return (
    <div className="space-y-4">
      <div className="bg-surface-2 rounded-lg p-4 border border-surface-3/50">
        <div className="flex items-center gap-2 mb-3">
          <Palette className="w-4 h-4 text-amber-400" />
          <span className="text-sm font-medium text-slate-300">Display</span>
        </div>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div>
            <div className="text-slate-500 mb-1">Assistant Name</div>
            <div className="text-slate-300">Lloyd</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Owner Display</div>
            <div className="text-slate-300 font-mono">raw</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Commands</div>
            <div className="text-slate-300 font-mono">auto</div>
          </div>
          <div>
            <div className="text-slate-500 mb-1">Ack Reaction Scope</div>
            <div className="text-slate-300 font-mono">group-mentions</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const [tab, setTab] = useState<Tab>("configuration");

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Settings className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Settings</h2>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-surface-3/30 pb-px">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-xs font-medium rounded-t-lg transition-colors ${
              tab === t.id
                ? "bg-surface-2 text-brand-400 border-b-2 border-brand-500"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === "configuration" && <ConfigurationTab />}
      {tab === "integrations" && <IntegrationsTab />}
      {tab === "agents" && <AgentsTab />}
      {tab === "interface" && <InterfaceTab />}
    </div>
  );
}
