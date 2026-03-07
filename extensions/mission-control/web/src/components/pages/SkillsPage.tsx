import { useEffect, useState, useCallback } from "react";
import { Sparkles, Package, Pencil, X, Save } from "lucide-react";
import { api, SkillInfo } from "../../api";
import { sanitizeHtml } from "../../utils/sanitize";

function ToggleSwitch({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
        enabled ? "bg-brand-600" : "bg-surface-3"
      }`}
      aria-label={enabled ? "Disable skill" : "Enable skill"}
    >
      <span
        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
          enabled ? "left-[14px]" : "left-[2px]"
        }`}
      />
    </button>
  );
}

export default function SkillsPage() {
  const [workspace, setWorkspace] = useState<SkillInfo[]>([]);
  const [bundled, setBundled] = useState<SkillInfo[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [saving, setSaving] = useState(false);

  const loadSkills = useCallback(() => {
    api
      .skills()
      .then((d) => {
        const ws = d.workspace || [];
        const bd = d.bundled || [];
        setWorkspace(ws);
        setBundled(bd);
        // Auto-select first skill if none selected
        setSelectedSkill((prev) => {
          if (prev) {
            // Update selected skill data from refreshed list
            const updated = [...ws, ...bd].find((s) => s.name === prev.name);
            return updated ?? prev;
          }
          return ws[0] ?? bd[0] ?? null;
        });
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    loadSkills();
  }, [loadSkills]);

  // Load skill content when selection changes
  useEffect(() => {
    if (!selectedSkill) return;
    setContent(null);
    setIsEditing(false);
    setLoadingContent(true);
    api
      .skillContent(selectedSkill.name)
      .then((d) => setContent(d.content))
      .catch(() => setContent(null))
      .finally(() => setLoadingContent(false));
  }, [selectedSkill?.name]);

  const handleToggle = async (skill: SkillInfo) => {
    const newEnabled = !skill.enabled;
    const updateList = (list: SkillInfo[]) =>
      list.map((s) => (s.name === skill.name ? { ...s, enabled: newEnabled } : s));
    setWorkspace(updateList);
    setBundled(updateList);
    setSelectedSkill((prev) =>
      prev?.name === skill.name ? { ...prev, enabled: newEnabled } : prev
    );
    try {
      await api.skillToggle(skill.name, newEnabled);
    } catch (err) {
      console.error("Skill toggle failed:", err);
      loadSkills();
    }
  };

  const handleEdit = () => {
    setEditContent(content ?? "");
    setIsEditing(true);
  };

  const handleCancel = () => {
    setIsEditing(false);
    setEditContent("");
  };

  const handleSave = async () => {
    if (!selectedSkill) return;
    setSaving(true);
    try {
      await api.skillContentSave(selectedSkill.name, editContent);
      setContent(editContent);
      setIsEditing(false);
    } catch (err) {
      console.error("Failed to save skill:", err);
    } finally {
      setSaving(false);
    }
  };

  const total = workspace.length + bundled.length;
  const enabledCount =
    workspace.filter((s) => s.enabled).length +
    bundled.filter((s) => s.enabled).length;

  const renderedContent = content
    ? sanitizeHtml(content)
    : "";

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left panel -- skill list */}
      <div className="w-1/4 min-w-48 flex flex-col border-r border-surface-3/50 overflow-hidden">
        {/* Header */}
        <div className="flex items-center gap-2 px-4 py-3 border-b border-surface-3/30 flex-shrink-0">
          <Sparkles className="w-4 h-4 text-brand-400" />
          <span className="text-sm font-semibold text-slate-200">Skills</span>
          <span className="ml-auto text-[10px] text-slate-500">
            {enabledCount}/{total}
          </span>
        </div>

        {/* Skill list */}
        <div className="flex-1 overflow-y-auto py-2">
          {workspace.length > 0 && (
            <div>
              <div className="px-4 py-1 text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                Workspace
              </div>
              {workspace.map((s) => (
                <button
                  key={s.name}
                  onClick={() => setSelectedSkill(s)}
                  className={`w-full flex items-center gap-2 px-4 py-2 text-left text-sm transition-colors ${
                    selectedSkill?.name === s.name
                      ? "bg-brand-600/20 text-slate-100"
                      : "hover:bg-surface-2/60 text-slate-400"
                  } ${!s.enabled ? "opacity-40" : ""}`}
                >
                  {s.emoji ? (
                    <span className="text-base leading-none flex-shrink-0">{s.emoji}</span>
                  ) : (
                    <Package className="w-3.5 h-3.5 flex-shrink-0 text-slate-500" />
                  )}
                  <span className="truncate text-xs">{s.name}</span>
                </button>
              ))}
            </div>
          )}

          {bundled.length > 0 && (
            <div className={workspace.length > 0 ? "mt-2" : ""}>
              <div className="px-4 py-1 text-[10px] uppercase tracking-wider text-slate-500 font-medium">
                Built-in
              </div>
              {bundled.map((s) => (
                <button
                  key={s.name}
                  onClick={() => setSelectedSkill(s)}
                  className={`w-full flex items-center gap-2 px-4 py-2 text-left text-sm transition-colors ${
                    selectedSkill?.name === s.name
                      ? "bg-brand-600/20 text-slate-100"
                      : "hover:bg-surface-2/60 text-slate-400"
                  } ${!s.enabled ? "opacity-40" : ""}`}
                >
                  {s.emoji ? (
                    <span className="text-base leading-none flex-shrink-0">{s.emoji}</span>
                  ) : (
                    <Package className="w-3.5 h-3.5 flex-shrink-0 text-slate-500" />
                  )}
                  <span className="truncate text-xs">{s.name}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Right panel -- skill detail */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {selectedSkill ? (
          <>
            {/* Detail header */}
            <div className="flex items-center gap-3 px-6 py-4 border-b border-surface-3/30 flex-shrink-0">
              {selectedSkill.emoji && (
                <span className="text-xl leading-none">{selectedSkill.emoji}</span>
              )}
              <h2 className="text-base font-semibold text-slate-100 flex-1">
                {selectedSkill.name}
              </h2>
              <ToggleSwitch
                enabled={selectedSkill.enabled}
                onToggle={() => handleToggle(selectedSkill)}
              />
            </div>

            {/* Content area */}
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {loadingContent ? (
                <div className="text-xs text-slate-500 animate-pulse">Loading...</div>
              ) : isEditing ? (
                <div className="space-y-3 h-full flex flex-col">
                  <textarea
                    value={editContent}
                    onChange={(e) => setEditContent(e.target.value)}
                    className="flex-1 min-h-64 w-full bg-surface-0 text-slate-200 text-xs font-mono rounded-lg p-3 border border-surface-3/50 resize-none focus:outline-none focus:border-brand-500/50"
                    spellCheck={false}
                  />
                  <div className="flex gap-2 justify-end flex-shrink-0">
                    <button
                      onClick={handleCancel}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-surface-2 rounded-lg transition-colors"
                    >
                      <X className="w-3 h-3" />
                      Cancel
                    </button>
                    <button
                      onClick={handleSave}
                      disabled={saving}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs text-white bg-brand-600 hover:bg-brand-500 rounded-lg disabled:opacity-50 transition-colors"
                    >
                      <Save className="w-3 h-3" />
                      {saving ? "Saving..." : "Save"}
                    </button>
                  </div>
                </div>
              ) : content ? (
                <div className="relative group">
                  <button
                    onClick={handleEdit}
                    className="absolute top-0 right-0 inline-flex items-center gap-1 px-2 py-1 text-[10px] text-slate-400 hover:text-slate-200 bg-surface-2 rounded opacity-0 group-hover:opacity-100 transition-opacity z-10"
                  >
                    <Pencil className="w-2.5 h-2.5" />
                    Edit
                  </button>
                  <div
                    className="prose-doc"
                    dangerouslySetInnerHTML={{ __html: renderedContent }}
                  />
                </div>
              ) : (
                <div className="text-xs text-slate-500 italic">
                  SKILL.md not found or could not be loaded.
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center text-slate-500 text-sm">
            Select a skill to view details
          </div>
        )}
      </div>
    </div>
  );
}
