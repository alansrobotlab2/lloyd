import { useEffect, useState, useCallback } from "react";
import { useReportMcFocus, usePendingFocusFor } from "../../contexts/McUiContext";
import { Sparkles, Search, RefreshCw } from "lucide-react";
import { Streamdown } from "streamdown";
import { api, type SkillInfo } from "../../api";
import { Input } from "@/components/ui/input";

function formatFrontmatter(content: string): string {
  if (!content.startsWith("---")) return content;
  const end = content.indexOf("\n---", 3);
  if (end === -1) return content;
  const yaml = content.slice(3, end).trim();
  const body = content.slice(end + 4).trimStart();
  return "```yaml\n" + yaml + "\n```\n\n" + body;
}

export default function SkillsPage() {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [selectedSkill, setSelectedSkill] = useState<SkillInfo | null>(null);
  const [content, setContent] = useState<string | null>(null);
  const [loadingContent, setLoadingContent] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  useReportMcFocus(
    "skills",
    selectedSkill ? { kind: "skill", id: selectedSkill.name } : null,
  );

  const pendingFocus = usePendingFocusFor("skills");
  useEffect(() => {
    if (!pendingFocus) return;
    const skill = skills.find((s) => s.name === pendingFocus);
    if (skill) setSelectedSkill(skill);
  }, [pendingFocus, skills]);

  const loadSkills = useCallback(() => {
    api.skills()
      .then((d) => {
        const ws = d.workspace || [];
        setSkills(ws);
        setSelectedSkill((prev) => {
          if (prev) {
            return ws.find((s) => s.name === prev.name) ?? prev;
          }
          return ws[0] ?? null;
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
    setLoadingContent(true);
    api
      .skillContent(selectedSkill.name)
      .then((d) => setContent(d.content))
      .catch(() => setContent(null))
      .finally(() => setLoadingContent(false));
  }, [selectedSkill?.name]);

  // Read-only page (#1293). The enable toggle and the editor's Save posted to
  // routes that were never registered; fetch resolves on a 404, so the toggle
  // flipped and the editor closed as if the SKILL.md had been written. Skills
  // are edited in the vault. Refresh is just the GET-backed list reload.
  const handleRefresh = () => {
    loadSkills();
  };

  const filtered = skills.filter(
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
      <div className="w-64 flex-shrink-0 border-r border-border flex flex-col">
        <div className="flex items-center gap-1.5 px-3 py-2.5 border-b border-border">
          <Sparkles size={13} className="text-primary" />
          <span className="text-xs font-medium text-foreground">Skills</span>
        </div>

        <div className="px-3 py-2 border-b border-border">
          <div className="relative">
            <Search size={13} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground pointer-events-none" />
            <Input
              type="text"
              placeholder="Search..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-7 text-xs"
            />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto py-1">
          {categories.map((cat) => (
            <div key={cat}>
              <div className="px-3 py-1.5 text-[10px] font-semibold text-foreground/90 uppercase tracking-wider bg-secondary border-y border-border/50 mt-1 first:mt-0">
                {cat}
              </div>
              {grouped[cat].map((skill) => (
                <button
                  key={skill.name}
                  onClick={() => setSelectedSkill(skill)}
                  className={`w-full text-left px-3 py-2 flex items-center gap-2 hover:bg-card transition-colors ${
                    selectedSkill?.name === skill.name ? "bg-card text-foreground" : skill.enabled ? "text-foreground/90" : "text-muted-foreground"
                  }`}
                >
                  <span className="text-xs flex-1 truncate">{skill.name}</span>
                  <span
                    className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${skill.enabled ? "bg-green-400" : "bg-muted"}`}
                  />
                </button>
              ))}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className="px-3 py-4 text-xs text-muted-foreground text-center">No skills found</div>
          )}
        </div>

        <div className="p-2 border-t border-border">
          <button
            onClick={handleRefresh}
            className="w-full flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground rounded-lg hover:bg-card transition-colors"
          >
            <RefreshCw size={12} />
            Refresh
          </button>
        </div>
      </div>

      {/* Detail pane */}
      {selectedSkill ? (
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-shrink-0 px-6 py-4 border-b border-border flex items-start justify-between gap-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <h2 className="text-base font-semibold text-foreground truncate">{selectedSkill.name}</h2>
              </div>
              {selectedSkill.description && (
                <p className="text-xs text-muted-foreground mt-0.5">{selectedSkill.description}</p>
              )}
              {selectedSkill.category && (
                <span className="mt-1 inline-block text-[10px] px-2 py-0.5 bg-secondary rounded text-muted-foreground">
                  {selectedSkill.category}
                </span>
              )}
            </div>
            <span className="flex-shrink-0 text-[10px] text-muted-foreground">
              Read-only — edit SKILL.md in the vault
            </span>
          </div>

          <div className="flex-1 overflow-y-auto px-6 py-4">
            {loadingContent && (
              <div className="text-xs text-muted-foreground">Loading...</div>
            )}
            {!loadingContent && content === null && (
              <div className="text-xs text-muted-foreground">No SKILL.md found for this skill.</div>
            )}
            {!loadingContent && content !== null && (
              <div className="prose-doc">
                <Streamdown>{formatFrontmatter(content)}</Streamdown>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
          Select a skill to view details
        </div>
      )}
    </div>
  );
}
