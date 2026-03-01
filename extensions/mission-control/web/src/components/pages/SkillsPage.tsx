import { useEffect, useState, useCallback } from "react";
import { Sparkles, Package, Tag, CheckCircle2, AlertTriangle } from "lucide-react";
import { api, SkillInfo } from "../../api";

function StatusBadge({ skill }: { skill: SkillInfo }) {
  if (!skill.configured) {
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] bg-amber-400/10 text-amber-400">
        <AlertTriangle className="w-2.5 h-2.5" />
        Missing deps
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full text-[10px] bg-emerald-400/10 text-emerald-400">
      <CheckCircle2 className="w-2.5 h-2.5" />
      Ready
    </span>
  );
}

function SkillCard({
  skill,
  onToggle,
}: {
  skill: SkillInfo;
  onToggle: (name: string, currentEnabled: boolean) => void;
}) {
  const bins = skill.requires?.bins ?? [];
  const env = skill.requires?.env ?? [];
  const config = skill.requires?.config ?? [];
  const hasDeps = bins.length > 0 || env.length > 0 || config.length > 0;

  return (
    <div
      className={`bg-surface-1 rounded-xl p-4 border transition-colors ${
        !skill.enabled
          ? "border-surface-3/30 opacity-50"
          : "border-surface-3/50 hover:border-brand-500/30"
      }`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <button
          onClick={() => onToggle(skill.name, skill.enabled)}
          className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
            skill.enabled ? "bg-brand-600" : "bg-surface-3"
          }`}
        >
          <span
            className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
              skill.enabled ? "left-[14px]" : "left-[2px]"
            }`}
          />
        </button>
        {skill.emoji ? (
          <span className="text-base leading-none">{skill.emoji}</span>
        ) : (
          <Package className="w-4 h-4 text-slate-500" />
        )}
        <span className="text-sm font-medium text-slate-200 truncate">
          {skill.name}
        </span>
        <StatusBadge skill={skill} />
      </div>
      <p className="text-xs text-slate-400 mb-2.5 line-clamp-2 leading-relaxed">
        {skill.description}
      </p>
      {hasDeps && (
        <div className="flex flex-wrap gap-1.5">
          {bins.map((b) => (
            <span
              key={b}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-surface-2 text-slate-400"
            >
              <Tag className="w-2.5 h-2.5" />
              {b}
            </span>
          ))}
          {env.map((e) => (
            <span
              key={e}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-amber-400/10 text-amber-400"
            >
              {e}
            </span>
          ))}
          {config.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-violet-400/10 text-violet-400"
            >
              {c}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default function SkillsPage() {
  const [workspace, setWorkspace] = useState<SkillInfo[]>([]);
  const [bundled, setBundled] = useState<SkillInfo[]>([]);

  const loadSkills = useCallback(() => {
    api
      .skills()
      .then((d) => {
        setWorkspace(d.workspace || []);
        setBundled(d.bundled || []);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    loadSkills();
  }, [loadSkills]);

  const handleToggle = async (skillName: string, currentEnabled: boolean) => {
    // Optimistic update
    const updateList = (list: SkillInfo[]) =>
      list.map((s) =>
        s.name === skillName ? { ...s, enabled: !currentEnabled } : s,
      );
    setWorkspace(updateList);
    setBundled(updateList);

    try {
      await api.skillToggle(skillName, !currentEnabled);
    } catch (err) {
      console.error("Skill toggle failed:", err);
      loadSkills(); // revert on error
    }
  };

  const total = workspace.length + bundled.length;
  const enabledCount =
    workspace.filter((s) => s.enabled).length +
    bundled.filter((s) => s.enabled).length;

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Sparkles className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Skills Explorer</h2>
        <span className="text-xs text-slate-500">
          {enabledCount} enabled / {total} total
        </span>
      </div>

      {/* Workspace skills */}
      {workspace.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Workspace
          </div>
          <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
            {workspace.map((s) => (
              <SkillCard key={s.name} skill={s} onToggle={handleToggle} />
            ))}
          </div>
        </div>
      )}

      {/* Built-in skills */}
      {bundled.length > 0 && (
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
            Built-in
          </div>
          <div className="grid grid-cols-2 xl:grid-cols-3 gap-3">
            {bundled.map((s) => (
              <SkillCard key={s.name} skill={s} onToggle={handleToggle} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
