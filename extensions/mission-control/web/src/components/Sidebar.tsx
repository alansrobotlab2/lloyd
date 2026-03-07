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
  Mic,
} from "lucide-react";
import { api } from "../api";

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
  | "settings";

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
  { id: "memory", label: "Memory", icon: Brain },
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
}

export default function Sidebar({ active, onNavigate, collapsed, onToggleCollapse }: SidebarProps) {
  const CollapseIcon = collapsed ? ChevronsRight : ChevronsLeft;
  const [workMode, setWorkMode] = useState(false);
  const [voiceActive, setVoiceActive] = useState(false);

  useEffect(() => {
    const poll = () => api.mode().then((s) => setWorkMode(s.currentMode === "work")).catch(() => {});
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const poll = () => fetch("/api/mc/voice-status")
      .then(r => r.json())
      .then(d => setVoiceActive(d.ws_active && d.has_client))
      .catch(() => setVoiceActive(false));
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
        {/* Voice Mode indicator */}
        {voiceActive && (
          <div
            title={collapsed ? "Voice Active" : undefined}
            className={`w-full flex items-center ${collapsed ? "justify-center" : "gap-2.5"} px-3 py-2 rounded-lg text-xs font-medium text-green-400 bg-green-600/10`}
          >
            <Mic className="w-4 h-4 flex-shrink-0 animate-pulse" />
            {!collapsed && <span className="truncate">Voice Active</span>}
          </div>
        )}
        {BOTTOM_ITEMS.map(renderItem)}
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
