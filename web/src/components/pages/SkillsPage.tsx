import { useEffect, useState, useCallback } from "react";
import { Sparkles, Package, Pencil, X, Save, Search, RefreshCw } from "lucide-react";
import { api, type SkillInfo } from "../../api";
import { sanitizeHtml } from "../../utils/sanitize";

function ToggleSwitch({ enabled, onToggle, disabled }: { enabled: boolean; onToggle: () => void; disabled?: boolean }) {
  return (
    <button
      onClick={onToggle}
      disabled={disabled}
      className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
        disabled ? "bg-surface-2 opacity-40 cursor-not-allowed" : enabled ? "bg-brand-600" : "bg-surface-3"
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
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"skills" | "builtin">("skills");

  const loadSkills = useCallback(() => {
    api.skills()
      .then((d) => {
        const ws = d.workspace || [];
        const bd = d.bundled || [];
        setWorkspace(ws);
        setBundled(bd);
        setSelectedSkill((prev) => {
          if (!prev) {
            if (ws.length === 0 && bd.length > 0) setActiveTab("builtin");
            else if (ws.length > 0) setActiveTab("skills");
          }
          return prev;
        });
        setSelectedSkill((prev) => {
          if (prev) {
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
    const patch = (list: SkillInfo[]) =>
      list.map((s) => (s.name === skill.name ? { ...s, enabled: newEnabled } : s));
    setWorkspace((prev) => patch(prev));
    setBundled((prev) => patch(prev));
    setSelectedSkill((prev) => (prev?.name === skill.name ? { ...prev, enabled: newEnabled } : prev));
    try {
      await api.skillToggle(skill.name, newEnabled);
    } catch {
      const revert = (list: SkillInfo[]) =>
        list.map((s) => (s.name === skill.name ? { ...s, enabled: skill.enabled } : s));
      setWorkspace((prev) => revert(prev));
      setBundled((prev) => revert(prev));
      setSelectedSkill((prev) => (prev?.name === skill.name ? { ...prev, enabled: skill.enabled } : prev));
    }
  };

  const handleSave = async () => {
    if (!selectedSkill) return;
    setSaving(true);
    try {
      await api.skillContentSave(selectedSkill.name, editContent);
      setContent(editContent);
      setIsEditing(false);
    } catch (e) {
      console.error("Save failed", e);
    } finally {
      setSaving(false);
    }
  };

  const handleRefresh = async () => {
    await api.skillsRefresh();
    loadSkills();
  };

  const activeList = activeTab === "skills" ? workspace : bundled;
  const filtered = activeList.filter(
    (s) =>
      !searchQuery ||
      s.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (s.description ?? "").toLowerCase().includes(searchQuery.toLowerCase()) ||
      (s.category ?? "").toLowerCase().includes(searchQuery.toLowerCase())
  );

  const grouped: Record<string, SkillInfo[]> = {};
  for (const skill of filtered) {
    const cat = skill.category ?? "Other";
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(skill);
  }
  const categories = Object.keys(grouped).sort();

  return (
    <div className="flex h-full overflow-hidden">
      {/* Sidebar */}
      <div className="w-64 flex-shrink-0 border-r border-slate-700 flex flex-col">
        <div className="flex border-b border-slate-700">
          <button
            onClick={() => setActiveTab("skills")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors ${
              activeTab === "skills"
                ? "text-brand-400 border-b-2 border-brand-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sparkles size={13} />
            Skills
          </button>
          <button
            onClick={() => setActiveTab("builtin")}
            className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors ${
              activeTab === "builtin"
                ? "text-brand-400 border-b-2 border-brand-400"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Package size={13} />
            Builtin
          </button>
        </div>

        <div className="px-3 py-2 border-b border-slate-700">
          <div className="flex items-center gap-2 bg-surface-1 rounded-lg px-2 py-1.5">
            <Search size={13} className="text-slate-400 flex-shrink-0" />
            <input
              type="text"
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-transparent text-xs text-slate-200 placeholder-slate-500 outline-none flex-1"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          {categories.map((cat) => (
            <div key={cat}>
              <div className="px-3 py-1.5 text-[10px] font-semibold text-slate-400 uppercase tracking-wider">
                {cat}
              </div>
              {grouped[cat].map((skill) => (
                <button
                  key={skill.name}
                  onClick={() => setSelectedSkill(skill)}
                  className={`w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-surface-1 transition-colors ${
                    selectedSkill?.name === skill.name ? "bg-surface-1 text-slate-200" : "text-slate-300"
                  }`}
                >
                  <span className="text-sm">{skill.emoji ?? "📦"}</span>
                  <span className="text-xs flex-1 truncate">{skill.name}</span>
                  <span
                    className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${skill.enabled ? "bg-green-400" : "bg-surface-3"}`}
                  />
                </button>
              ))}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="px-3 py-4 text-xs text-slate-400 text-center">No skills found</div>
          )}
        </div>

        <div className="p-2 border-t border-slate-700">
          <button
            onClick={handleRefresh}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 rounded-lg hover:bg-surface-1 transition-colors"
          >
            <RefreshCw size={12} />
            Refresh
          </button>
        </div>
      </div>

      {/* Detail pane */}
      {selectedSkill ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-shrink-0 px-6 py-4 border-b border-slate-700 flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xl">{selectedSkill.emoji ?? "📦"}</span>
                <h2 className="text-base font-semibold text-slate-200 truncate">{selectedSkill.name}</h2>
              </div>
              {selectedSkill.description && (
                <p className="text-xs text-slate-400 mt-0.5">{selectedSkill.description}</p>
              )}
              {selectedSkill.category && (
                <span className="mt-1 inline-block text-[10px] px-2 py-0.5 bg-surface-2 rounded text-slate-400">
                  {selectedSkill.category}
                </span>
              )}
            </div>
            <div className="flex items-center gap-3 flex-shrink-0">
              <ToggleSwitch
                enabled={selectedSkill.enabled}
                onToggle={() => handleToggle(selectedSkill)}
              />
              {content !== null && !isEditing && (
                <button
                  onClick={() => { setEditContent(content); setIsEditing(true); }}
                  className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-200 rounded-lg hover:bg-surface-1 transition-colors"
                >
                  <Pencil size={12} />
                  Edit
                </button>
              )}
              {isEditing && (
                <>
                  <button
                    onClick={handleSave}
                    disabled={saving}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs bg-brand-600 hover:bg-brand-500 text-white rounded-lg transition-colors disabled:opacity-50"
                  >
                    <Save size={12} />
                    {saving ? "Saving..." : "Save"}
                  </button>
                  <button
                    onClick={() => setIsEditing(false)}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs text-slate-400 hover:text-slate-200 rounded-lg hover:bg-surface-1 transition-colors"
                  >
                    <X size={12} />
                    Cancel
                  </button>
                </>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4">
            {loadingContent && (
              <div className="text-xs text-slate-400">Loading...</div>
            )}
            {!loadingContent && content === null && (
              <div className="text-xs text-slate-400">No SKILL.md found for this skill.</div>
            )}
            {!loadingContent && content !== null && isEditing && (
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-full min-h-[400px] bg-surface-0 border border-slate-700 rounded-lg p-3 text-xs font-mono text-slate-200 resize-none outline-none focus:border-brand-500"
              />
            )}
            {!loadingContent && content !== null && !isEditing && (
              <div
                className="prose-doc"
                dangerouslySetInnerHTML={{ __html: sanitizeHtml(content) }}
              />
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-slate-400 text-sm">
          Select a skill to view details
        </div>
      )}
    </div>
  );
}
