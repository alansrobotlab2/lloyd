import { X, Plus } from "lucide-react";
import type { TabState } from "./hooks/useSessionTabs";

interface SessionTabBarProps {
  tabs: TabState[];
  activeTabId: string | null;
  onSelect: (tabId: string) => void;
  onClose: (tabId: string) => void;
  onNew: () => void;
}

export default function SessionTabBar({
  tabs,
  activeTabId,
  onSelect,
  onClose,
  onNew,
}: SessionTabBarProps) {
  return (
    <div className="flex items-center gap-1 px-3 py-1.5 border-b border-surface-3/50 bg-surface-1/50 overflow-x-auto scrollbar-hide">
      {tabs.map((tab) => {
        const isActive = tab.id === activeTabId;
        return (
          <button
            key={tab.id}
            onClick={() => onSelect(tab.id)}
            className={`group flex items-center gap-1.5 px-3 py-1.5 rounded-t-lg text-xs cursor-pointer transition-colors flex-shrink-0 ${
              isActive
                ? "bg-surface-1 text-slate-200 border border-surface-3/50 border-b-transparent"
                : "text-slate-500 hover:text-slate-300 hover:bg-surface-2/50"
            }`}
          >
            <span className="truncate max-w-[20ch]">{tab.label}</span>
            {tab.thinking && (
              <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse flex-shrink-0" />
            )}
            <span
              onClick={(e) => {
                e.stopPropagation();
                onClose(tab.id);
              }}
              className={`flex-shrink-0 rounded p-0.5 transition-colors ${
                isActive
                  ? "text-slate-400 hover:text-slate-200 hover:bg-surface-3/50"
                  : "text-transparent group-hover:text-slate-500 hover:!text-slate-300 hover:bg-surface-3/50"
              }`}
            >
              <X className="w-3 h-3" />
            </span>
          </button>
        );
      })}
      <button
        onClick={onNew}
        title="New session"
        className="flex-shrink-0 text-slate-500 hover:text-brand-400 hover:bg-surface-2/50 rounded-lg p-1.5 transition-colors"
      >
        <Plus className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}
