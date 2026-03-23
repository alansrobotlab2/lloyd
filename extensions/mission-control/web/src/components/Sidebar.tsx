import { useState, useEffect } from "react";
import {
  Activity,
  ChartArea,
  LayoutList,
  Brain,
  Sparkles,
  Users,
  Cpu,
  Clock,
  Wrench,
  LayoutGrid,
  Settings,
  Bot,
  MessageCircle,
  ChevronsLeft,
  ChevronsRight,
  Briefcase,
  GitBranch,
  Lightbulb,
  Code2,
  Mic,
  MicOff,
  Power,
  Radio,
  Volume2,
  VolumeX,
} from "lucide-react";
import { api } from "../api";
import { useVoiceContext } from "../contexts/VoiceContext";

export type Page =
  | "chat"
  | "services"
  | "dashboard"
  | "backlog"
  | "memory"
  | "graph"
  | "skills"
  | "sessions"
  | "agents"
  | "models"
  | "cron"
  | "tools"
  | "activity"
  | "settings"
  | "autonomy"
  | "architecture";

interface NavItem {
  id: Page;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { id: "chat", label: "Chat", icon: MessageCircle },
  { id: "activity", label: "Activity", icon: Activity },
  { id: "dashboard", label: "Usage", icon: ChartArea },
  { id: "backlog", label: "Backlog", icon: LayoutGrid },
  { id: "autonomy", label: "Autonomy", icon: Lightbulb },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "architecture", label: "Architecture", icon: Code2 },
  { id: "skills", label: "Skills", icon: Sparkles },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "sessions", label: "Sessions", icon: Users },
  { id: "agents", label: "Agents", icon: Bot },
  { id: "models", label: "Models", icon: Cpu },
  { id: "cron", label: "Crontab", icon: Clock },
  { id: "services", label: "Services", icon: LayoutList },
];

const BOTTOM_ITEMS: NavItem[] = [
  { id: "settings", label: "Settings", icon: Settings },
];

interface SidebarProps {
  active: Page;
  onNavigate: (page: Page) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  sessionKey?: string | null;
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse, sessionKey }: SidebarProps) {
  const CollapseIcon = collapsed ? ChevronsRight : ChevronsLeft;
  const [workMode, setWorkMode] = useState(false);
  const { isListening, voiceEnabled, wsAvailable, statusLoaded, latestTranscript, transcriptVisible, stateColor, stateText, startMic, stopMic, handleVoiceToggle, ttsEnabled, handleTtsToggle, pipelineState } = useVoiceContext();

  useEffect(() => {
    const poll = () => api.mode().then((s) => setWorkMode(s.currentMode === "work")).catch(() => {});
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const renderItem = (item: NavItem) => {
    const Icon = item.icon;
    const isActive = active === item.id;
    return (
      <button
        key={item.id}
        onClick={() => onNavigate(item.id)}
        title={collapsed ? item.label : undefined}
        className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
          isActive
            ? "bg-brand-600/15 text-brand-400"
            : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
        }`}
      >
        <Icon className="w-4 h-4 flex-shrink-0" />
        {!collapsed && <span className="truncate">{item.label}</span>}
      </button>
    );
  };

  return (
    <aside className={`${collapsed ? "w-14" : "w-48"} bg-surface-1 border-r border-surface-3/30 flex flex-col py-4 transition-all duration-200`}>
      {/* Brand */}
      <div className={`${collapsed ? "px-2 justify-center" : "px-4"} mb-6 flex items-center gap-2`}>
        <img src="/api/mc/agent-avatar?id=lloyd" alt="Lloyd" className="w-7 h-7 rounded-lg object-cover flex-shrink-0" />
        {!collapsed && (
          <div>
            <div className="text-sm font-bold tracking-wide">LLOYD</div>
            <div className="text-[10px] text-slate-500 -mt-0.5">Mission Control</div>
          </div>
        )}
      </div>

      {/* Main nav */}
      <nav className="flex-1 px-2 space-y-0.5">
        {NAV_ITEMS.map(renderItem)}
      </nav>

      {/* Bottom nav */}
      <div className="px-2 pt-2 border-t border-surface-3/30 space-y-0.5">
        {/* Voice status section */}
        <div className="space-y-0.5 mb-1">
          {!collapsed && (
            <div className="px-3 pt-1 pb-0.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500">Voice Mode</div>
          )}
          {/* Row 1 — Last utterance */}
          {latestTranscript && (
            <div
              title={collapsed ? latestTranscript : undefined}
              className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-1.5 rounded-lg text-xs transition-opacity duration-1000 ${transcriptVisible ? "opacity-100" : "opacity-0"} text-slate-400`}
            >
              {collapsed ? (
                <MessageCircle className="w-4 h-4 flex-shrink-0 text-slate-500" />
              ) : (
                <span className="truncate text-[11px] italic">"{latestTranscript}"</span>
              )}
            </div>
          )}

          {/* Row 2 — Pipeline state */}
          {wsAvailable && (
            <div
              title={collapsed ? stateText : undefined}
              className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-1.5 rounded-lg text-xs font-medium ${stateColor}`}
            >
              <Radio className="w-4 h-4 flex-shrink-0" />
              {!collapsed && <span className="truncate text-[11px] font-mono">{stateText}</span>}
            </div>
          )}

          {/* Row 3 — Mic toggle */}
          <button
            onClick={wsAvailable && statusLoaded ? () => (isListening ? stopMic() : startMic(sessionKey || undefined)) : undefined}
            disabled={!wsAvailable || !statusLoaded}
            title={collapsed ? (isListening ? "Mic Active" : "Mic Inactive") : undefined}
            className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              !wsAvailable || !statusLoaded
                ? "text-slate-600 cursor-not-allowed opacity-50"
                : isListening
                  ? "text-green-400 bg-green-600/10 hover:bg-green-600/20"
                  : "text-red-400 bg-red-600/10 hover:bg-red-600/20"
            }`}
          >
            {isListening ? <Mic className={`w-4 h-4 flex-shrink-0 ${pipelineState === "LISTENING" || pipelineState === "ACTIVE_LISTEN" ? "animate-pulse" : ""}`} /> : <MicOff className="w-4 h-4 flex-shrink-0" />}
            {!collapsed && <span className="truncate">{isListening ? "Active" : "Inactive"}</span>}
          </button>

          {/* Row 3.5 — Speaker toggle */}
          <button
            onClick={handleTtsToggle}
            title={collapsed ? (ttsEnabled ? "Voice On" : "Voice Off") : undefined}
            className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              ttsEnabled
                ? "text-green-400 bg-green-600/10 hover:bg-green-600/20"
                : "text-red-400 bg-red-600/10 hover:bg-red-600/20"
            }`}
          >
            {ttsEnabled ? <Volume2 className="w-4 h-4 flex-shrink-0" /> : <VolumeX className="w-4 h-4 flex-shrink-0" />}
            {!collapsed && <span className="truncate">{ttsEnabled ? "Voice" : "Voice"}</span>}
          </button>

          {/* Row 4 — Power toggle */}
          <button
            onClick={statusLoaded ? handleVoiceToggle : undefined}
            disabled={!statusLoaded}
            title={collapsed ? (voiceEnabled ? "Voice Enabled" : "Voice Disabled") : undefined}
            className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
              !statusLoaded
                ? "text-slate-600 cursor-not-allowed opacity-50"
                : voiceEnabled
                  ? "text-green-400 bg-green-600/10 hover:bg-green-600/20"
                  : "text-slate-500 hover:text-slate-300 hover:bg-surface-2"
            }`}
          >
            <Power className="w-4 h-4 flex-shrink-0" />
            {!collapsed && <span className="truncate">{voiceEnabled ? "Enabled" : "Disabled"}</span>}
          </button>
        </div>
        <div className="border-t border-surface-3/30 my-1" />

        {/* Work Mode toggle */}
        <button
          onClick={async () => {
            const next = !workMode;
            setWorkMode(next);
            try {
              const result = await api.modeSet(next ? "work" : "personal");
              setWorkMode(result.currentMode === "work");
            } catch {
              setWorkMode(!next);
            }
          }}
          title={collapsed ? "Work Mode" : undefined}
          className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium transition-colors ${
            workMode
              ? "bg-purple-600 text-white hover:bg-purple-500"
              : "text-slate-400 hover:text-slate-200 hover:bg-surface-2"
          }`}
        >
          <Briefcase className="w-4 h-4 flex-shrink-0" />
          {!collapsed && (
            <span className="truncate flex-1 text-left">Work Mode</span>
          )}
          {!collapsed && (
            <span className={`w-4 h-4 rounded border-2 flex-shrink-0 flex items-center justify-center transition-colors ${
              workMode ? "bg-white border-white" : "border-slate-500"
            }`}>
              {workMode && <span className="text-[10px] text-purple-600 font-bold leading-none">✓</span>}
            </span>
          )}
        </button>
        <div className="border-t border-surface-3/30 my-1" />
        {BOTTOM_ITEMS.map(renderItem)}
        <div className="border-t border-surface-3/30 my-1" />
        <button
          onClick={onToggleCollapse}
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          className="w-full flex items-center justify-center px-3 py-2 rounded-lg text-xs font-medium text-slate-500 hover:text-slate-300 hover:bg-surface-2 transition-colors mt-1"
        >
          <CollapseIcon className="w-4 h-4" />
        </button>
      </div>
    </aside>
  );
}
