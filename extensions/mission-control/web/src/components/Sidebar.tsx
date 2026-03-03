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
} from "lucide-react";

export type Page =
  | "chat"
  | "services"
  | "dashboard"
  | "clawdeck"
  | "memory"
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
  { id: "clawdeck", label: "ClawDeck", icon: LayoutGrid },
  { id: "memory", label: "Memory", icon: Brain },
  { id: "skills", label: "Skills Explorer", icon: Sparkles },
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
