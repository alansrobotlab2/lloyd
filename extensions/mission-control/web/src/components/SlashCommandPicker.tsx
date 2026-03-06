import { useEffect, useRef } from "react";
import { Terminal, Sliders, Info, Shield, Wrench, Sparkles, Puzzle, Music } from "lucide-react";
import type { CommandInfo } from "../api";

const CATEGORY_ORDER = ["session", "options", "status", "management", "tools", "media", "plugin", "skill"];

const CATEGORY_META: Record<string, { label: string; Icon: typeof Terminal }> = {
  session: { label: "Session", Icon: Terminal },
  options: { label: "Options", Icon: Sliders },
  status: { label: "Status", Icon: Info },
  management: { label: "Management", Icon: Shield },
  tools: { label: "Tools", Icon: Wrench },
  media: { label: "Media", Icon: Music },
  plugin: { label: "Plugins", Icon: Puzzle },
  skill: { label: "Skills", Icon: Sparkles },
};

interface Props {
  commands: CommandInfo[];
  filter: string;
  selectedIndex: number;
  onSelect: (cmd: CommandInfo) => void;
  onHover: (index: number) => void;
}

export default function SlashCommandPicker({ commands, filter, selectedIndex, onSelect, onHover }: Props) {
  const listRef = useRef<HTMLDivElement>(null);

  const filtered = commands.filter((cmd) => {
    if (!filter) return true;
    const f = filter.toLowerCase();
    return cmd.name.toLowerCase().includes(f) || cmd.description.toLowerCase().includes(f);
  });

  // Group by category, preserving order
  const grouped = CATEGORY_ORDER
    .map((cat) => ({
      category: cat,
      ...(CATEGORY_META[cat] || { label: cat, Icon: Terminal }),
      items: filtered.filter((c) => c.category === cat),
    }))
    .filter((g) => g.items.length > 0);

  // Flat list for index mapping
  const flatItems = grouped.flatMap((g) => g.items);

  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${selectedIndex}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  if (flatItems.length === 0) return null;

  let globalIdx = 0;

  return (
    <div
      ref={listRef}
      className="absolute bottom-full left-0 right-0 mb-1 bg-surface-2 border border-surface-3 rounded-lg shadow-xl max-h-72 overflow-y-auto z-50"
    >
      {grouped.map((group) => (
        <div key={group.category}>
          <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-slate-500 bg-surface-1/50 sticky top-0 flex items-center gap-1.5">
            <group.Icon className="w-3 h-3" />
            {group.label}
          </div>
          {group.items.map((cmd) => {
            const idx = globalIdx++;
            const isSelected = idx === selectedIndex;
            return (
              <button
                key={cmd.name}
                data-idx={idx}
                onMouseEnter={() => onHover(idx)}
                onMouseDown={(e) => {
                  e.preventDefault(); // prevent input blur
                  onSelect(cmd);
                }}
                className={`w-full text-left px-3 py-2 flex items-center gap-3 text-sm transition-colors ${
                  isSelected
                    ? "bg-brand-600/20 text-slate-100"
                    : "text-slate-300 hover:bg-surface-3/30"
                }`}
              >
                <span className="text-brand-400 font-mono text-xs min-w-[120px]">/{cmd.name}</span>
                <span className="text-slate-500 text-xs truncate">{cmd.description}</span>
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}
