import { useEffect, useState, useCallback } from "react";
import { Bot, ChevronLeft, ChevronDown, ChevronRight, Cpu, Wrench, Sparkles, Users, Layers, FileText, Pencil, X, Save } from "lucide-react";
import { marked } from "marked";
import { api, AgentInfo, AgentsData, AgentStatusData, SubagentRunInfo, ToolGroupInfo, WorkspaceFile } from "../../api";

// ── Helpers ─────────────────────────────────────────────────────────────

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

const STATE_COLORS: Record<string, string> = {
  idle: "bg-slate-500",
  processing: "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]",
  waiting: "bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.5)]",
};

const STATE_BADGES: Record<string, string> = {
  idle: "bg-slate-500/20 text-slate-400 border-slate-500/30",
  processing: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  waiting: "bg-amber-500/20 text-amber-400 border-amber-500/30",
};

function outcomeBadgeClass(outcome?: string): string {
  switch (outcome) {
    case "ok":
      return "bg-emerald-500/20 text-emerald-400 border-emerald-500/30";
    case "error":
      return "bg-red-500/20 text-red-400 border-red-500/30";
    case "timeout":
      return "bg-amber-500/20 text-amber-400 border-amber-500/30";
    default:
      return "bg-slate-500/20 text-slate-400 border-slate-500/30";
  }
}

// ── Toggle Switch (reusable) ────────────────────────────────────────────

function Toggle({
  checked,
  disabled,
  onChange,
}: {
  checked: boolean;
  disabled?: boolean;
  onChange: () => void;
}) {
  return (
    <button
      onClick={disabled ? undefined : onChange}
      className={`relative w-8 h-[18px] rounded-full transition-colors flex-shrink-0 ${
        disabled ? "opacity-40 cursor-default" : "cursor-pointer"
      } ${checked ? "bg-brand-600" : "bg-surface-3"}`}
    >
      <span
        className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-white transition-transform ${
          checked ? "left-[14px]" : "left-[2px]"
        }`}
      />
    </button>
  );
}

// ── Agent Live Status Banner ────────────────────────────────────────────

function AgentLiveStatus({ status }: { status: AgentStatusData | null }) {
  if (!status) {
    return (
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 animate-pulse">
        <div className="h-4 bg-surface-2 rounded w-32" />
      </div>
    );
  }

  const { mainAgent, activity } = status;
  const stateColor = STATE_COLORS[mainAgent.state] ?? STATE_COLORS.idle;
  const stateBadge = STATE_BADGES[mainAgent.state] ?? STATE_BADGES.idle;

  return (
    <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
      <div className="flex items-center gap-3">
        <div className="relative flex-shrink-0">
          <div className={`w-3 h-3 rounded-full ${stateColor}`} />
          {mainAgent.state === "processing" && (
            <div className="absolute inset-0 w-3 h-3 rounded-full bg-emerald-400 animate-ping opacity-40" />
          )}
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-mono border ${stateBadge}`}>
          {mainAgent.state}
        </span>
        <div className="flex-1 text-xs text-slate-400">
          {activity.label}
          {activity.detail && (
            <span className="text-slate-500 font-mono ml-1.5">{activity.detail}</span>
          )}
        </div>
        {activity.type !== "idle" && (
          <span className="text-[10px] text-slate-600 font-mono">
            {formatElapsed(activity.elapsedMs)}
          </span>
        )}
        {mainAgent.queueDepth > 0 && (
          <span className="text-[10px] text-amber-400 font-mono">
            {mainAgent.queueDepth} queued
          </span>
        )}
      </div>
      {status.heartbeat && (status.heartbeat.active > 0 || status.heartbeat.waiting > 0 || status.heartbeat.queued > 0) && (
        <div className="flex gap-4 mt-2 text-[10px] text-slate-500 font-mono">
          <span>{status.heartbeat.active} active</span>
          <span>{status.heartbeat.waiting} waiting</span>
          <span>{status.heartbeat.queued} queued</span>
        </div>
      )}
    </div>
  );
}

// ── Subagent Run Card ───────────────────────────────────────────────────

function SubagentRunCard({ run, isActive }: { run: SubagentRunInfo; isActive: boolean }) {
  const elapsed = isActive
    ? Date.now() - (run.startedAt ?? run.createdAt)
    : run.durationMs;

  return (
    <div
      className={`bg-surface-1 rounded-xl px-4 py-3 border transition-colors ${
        isActive ? "border-emerald-500/30" : "border-surface-3/50"
      }`}
    >
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0">
          {isActive ? (
            <div className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]" />
          ) : (
            <div
              className={`w-2 h-2 rounded-full ${
                run.outcome === "ok" ? "bg-emerald-400" : "bg-red-400"
              }`}
            />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-xs text-slate-300 line-clamp-2">{run.task}</div>
          <div className="flex items-center gap-2 mt-1 flex-wrap">
            {run.label && (
              <span className="text-[10px] text-slate-500 font-mono">{run.label}</span>
            )}
            {run.model && (
              <span className="text-[10px] text-slate-500 font-mono">{run.model}</span>
            )}
            {run.spawnMode && (
              <span className="text-[10px] text-slate-600 font-mono">{run.spawnMode}</span>
            )}
          </div>
        </div>
        <div className="text-right flex-shrink-0">
          {elapsed != null && (
            <div className="text-[10px] text-slate-500 font-mono">
              {formatElapsed(elapsed)}
            </div>
          )}
          {!isActive && run.outcome && (
            <span
              className={`text-[10px] px-1.5 py-0.5 rounded-full font-mono border mt-0.5 inline-block ${outcomeBadgeClass(run.outcome)}`}
            >
              {run.outcome}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Subagent Runs List ──────────────────────────────────────────────────

function SubagentRunsList({ status }: { status: AgentStatusData }) {
  const { active, recentCompleted } = status.subagents;

  if (active.length === 0 && recentCompleted.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic py-4 text-center">
        No subagent runs
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {active.length > 0 && (
        <>
          <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-medium flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
            Active ({active.length})
          </div>
          {active.map((run) => (
            <SubagentRunCard key={run.runId} run={run} isActive />
          ))}
        </>
      )}

      {recentCompleted.length > 0 && (
        <>
          <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mt-3">
            Recent
          </div>
          {recentCompleted.map((run) => (
            <SubagentRunCard key={run.runId} run={run} isActive={false} />
          ))}
        </>
      )}
    </div>
  );
}

// ── Agent Card (list view) ──────────────────────────────────────────────

function AgentCard({
  agent,
  agentState,
  onClick,
}: {
  agent: AgentInfo;
  agentState?: "idle" | "processing" | "waiting";
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="w-full bg-surface-1 rounded-xl p-5 border border-surface-3/50 hover:border-brand-500/30 transition-colors text-left"
    >
      <div className="flex items-center gap-4">
        {agent.avatar ? (
          <img
            src={`/mc/${agent.avatar.replace("avatars/", "")}`}
            alt={agent.id}
            className="w-12 h-12 rounded-xl object-cover flex-shrink-0"
          />
        ) : (
          <div className="w-12 h-12 rounded-xl bg-surface-2 flex items-center justify-center flex-shrink-0">
            <Bot className="w-6 h-6 text-slate-500" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-200">{agent.name ?? agent.id}</span>
            {agentState && agentState !== "idle" && (
              <div className={`w-2 h-2 rounded-full flex-shrink-0 ${STATE_COLORS[agentState]}`} />
            )}
          </div>
          {agent.primaryModel && (
            <div className="text-[11px] text-slate-500 font-mono mt-0.5 truncate">
              {agent.primaryModel}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mt-4">
        <Stat icon={Users} label="sessions" value={agent.sessions.active} sub={`/ ${agent.sessions.total}`} />
        <Stat icon={Cpu} label="models" value={agent.enabledModels} sub={`/ ${agent.modelCount}`} />
        <Stat icon={Wrench} label="tools disabled" value={agent.disabledTools} />
        <Stat icon={Layers} label="subagents" value={agent.subagentMaxConcurrent ?? 0} sub="max" />
      </div>
    </button>
  );
}

function Stat({
  icon: Icon,
  label,
  value,
  sub,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number;
  sub?: string;
}) {
  return (
    <div className="text-center">
      <Icon className="w-3.5 h-3.5 mx-auto text-slate-500 mb-1" />
      <div className="text-sm font-mono text-slate-300">
        {value}
        {sub && <span className="text-slate-600 text-[10px] ml-0.5">{sub}</span>}
      </div>
      <div className="text-[10px] text-slate-500">{label}</div>
    </div>
  );
}

// ── Agent Tools Editor ──────────────────────────────────────────────────

function AgentToolsEditor({
  agent,
  allToolGroups,
  onUpdated,
}: {
  agent: AgentInfo;
  allToolGroups: ToolGroupInfo[];
  onUpdated: () => void;
}) {
  const [toolsAllow, setToolsAllow] = useState<string[] | null>(agent.toolsAllow);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Reset when agent changes
  useEffect(() => {
    setToolsAllow(agent.toolsAllow);
    setDirty(false);
  }, [agent.id, agent.toolsAllow]);

  const isInherited = toolsAllow === null;
  const allowSet = new Set(toolsAllow ?? []);
  const allTools = [...new Set(allToolGroups.flatMap((g) => g.tools))];
  const allowCount = isInherited ? allTools.length : toolsAllow!.length;

  // Track which tool names appear in multiple groups
  const toolSourceCount = new Map<string, number>();
  for (const g of allToolGroups) {
    for (const t of g.tools) {
      toolSourceCount.set(t, (toolSourceCount.get(t) ?? 0) + 1);
    }
  }

  const handleToggle = (toolName: string) => {
    if (isInherited) {
      // Switch from inherited to explicit: start with all, remove this one
      setToolsAllow(allTools.filter((t) => t !== toolName));
    } else {
      if (allowSet.has(toolName)) {
        setToolsAllow(toolsAllow!.filter((t) => t !== toolName));
      } else {
        setToolsAllow([...toolsAllow!, toolName]);
      }
    }
    setDirty(true);
  };

  const handleInheritToggle = () => {
    if (isInherited) {
      setToolsAllow([...allTools]);
    } else {
      setToolsAllow(null);
    }
    setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.agentToolsUpdate(agent.id, toolsAllow);
      setDirty(false);
      onUpdated();
    } catch (err) {
      console.error("Failed to save tools:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setToolsAllow(agent.toolsAllow);
    setDirty(false);
  };

  return (
    <div className="space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-slate-500" />
        ) : (
          <ChevronRight className="w-3 h-3 text-slate-500" />
        )}
        <Wrench className="w-3.5 h-3.5 text-slate-500" />
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Allowed Tools
        </span>
        <span className="text-[10px] text-slate-600">
          {isInherited ? "all (inherited)" : `${allowCount} / ${allTools.length}`}
        </span>
      </button>

      {expanded && (
        <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 space-y-4">
          {/* Inherit toggle */}
          <div className="flex items-center gap-3 pb-3 border-b border-surface-3/30">
            <Toggle checked={isInherited} onChange={handleInheritToggle} />
            <span className="text-xs text-slate-300">Inherit all tools</span>
            {isInherited && (
              <span className="text-[10px] text-slate-500 italic">
                Agent has access to all available tools
              </span>
            )}
          </div>

          {/* Tool groups */}
          <div className="space-y-4">
            {allToolGroups.map((group) => (
              <div key={group.source} className="space-y-2">
                <div className="text-[10px] text-slate-500 font-medium">{group.source}</div>
                <div className="grid grid-cols-2 xl:grid-cols-3 gap-2">
                  {group.tools.map((tool) => {
                    const checked = isInherited || allowSet.has(tool);
                    const isShared = (toolSourceCount.get(tool) ?? 1) > 1;
                    return (
                      <div
                        key={tool}
                        className={`bg-surface-0 rounded-lg px-3 py-2 border transition-colors flex items-center gap-3 ${
                          checked ? "border-surface-3/50" : "border-surface-3/30 opacity-50"
                        }`}
                      >
                        <Toggle
                          checked={checked}
                          disabled={isInherited}
                          onChange={() => handleToggle(tool)}
                        />
                        <span className="text-xs font-mono text-slate-300 truncate">{tool}</span>
                        {isShared && (
                          <span className="text-[9px] text-amber-500/70 flex-shrink-0" title="This tool name exists in multiple sources — toggling affects all instances">
                            shared
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {/* Save / Cancel */}
          {dirty && (
            <div className="flex gap-2 justify-end pt-3 border-t border-surface-3/30">
              <button
                onClick={handleCancel}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-surface-2 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-xs text-white bg-brand-600 hover:bg-brand-500 rounded-lg disabled:opacity-50 transition-colors"
              >
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Agent Skills Editor ─────────────────────────────────────────────────

function AgentSkillsEditor({
  agent,
  allSkillNames,
  onUpdated,
}: {
  agent: AgentInfo;
  allSkillNames: string[];
  onUpdated: () => void;
}) {
  const [skills, setSkills] = useState<string[] | null>(agent.skills);
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    setSkills(agent.skills);
    setDirty(false);
  }, [agent.id, agent.skills]);

  const isInherited = skills === null;
  const skillSet = new Set(skills ?? []);
  const allowCount = isInherited ? allSkillNames.length : skills!.length;

  const handleToggle = (skillName: string) => {
    if (isInherited) {
      setSkills(allSkillNames.filter((s) => s !== skillName));
    } else {
      if (skillSet.has(skillName)) {
        setSkills(skills!.filter((s) => s !== skillName));
      } else {
        setSkills([...skills!, skillName]);
      }
    }
    setDirty(true);
  };

  const handleInheritToggle = () => {
    if (isInherited) {
      setSkills([...allSkillNames]);
    } else {
      setSkills(null);
    }
    setDirty(true);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.agentSkillsUpdate(agent.id, skills);
      setDirty(false);
      onUpdated();
    } catch (err) {
      console.error("Failed to save skills:", err);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    setSkills(agent.skills);
    setDirty(false);
  };

  if (allSkillNames.length === 0) return null;

  return (
    <div className="space-y-1">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-2 w-full text-left"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-slate-500" />
        ) : (
          <ChevronRight className="w-3 h-3 text-slate-500" />
        )}
        <Sparkles className="w-3.5 h-3.5 text-slate-500" />
        <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Skills
        </span>
        <span className="text-[10px] text-slate-600">
          {isInherited ? "all (inherited)" : `${allowCount} / ${allSkillNames.length}`}
        </span>
      </button>

      {expanded && (
        <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 space-y-4">
          {/* Inherit toggle */}
          <div className="flex items-center gap-3 pb-3 border-b border-surface-3/30">
            <Toggle checked={isInherited} onChange={handleInheritToggle} />
            <span className="text-xs text-slate-300">Inherit all skills</span>
            {isInherited && (
              <span className="text-[10px] text-slate-500 italic">
                Agent has access to all available skills
              </span>
            )}
          </div>

          {/* Skills list */}
          <div className="grid grid-cols-2 xl:grid-cols-3 gap-2">
            {allSkillNames.map((skill) => {
              const checked = isInherited || skillSet.has(skill);
              return (
                <div
                  key={skill}
                  className={`bg-surface-0 rounded-lg px-3 py-2 border transition-colors flex items-center gap-3 ${
                    checked ? "border-surface-3/50" : "border-surface-3/30 opacity-50"
                  }`}
                >
                  <Toggle
                    checked={checked}
                    disabled={isInherited}
                    onChange={() => handleToggle(skill)}
                  />
                  <span className="text-xs font-mono text-slate-300 truncate">{skill}</span>
                </div>
              );
            })}
          </div>

          {/* Save / Cancel */}
          {dirty && (
            <div className="flex gap-2 justify-end pt-3 border-t border-surface-3/30">
              <button
                onClick={handleCancel}
                className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-200 bg-surface-2 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="px-3 py-1.5 text-xs text-white bg-brand-600 hover:bg-brand-500 rounded-lg disabled:opacity-50 transition-colors"
              >
                {saving ? "Saving..." : "Save"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Workspace File Editor ───────────────────────────────────────────────

function WorkspaceFileEditor({
  agentId,
  file,
  onSaved,
}: {
  agentId: string;
  file: WorkspaceFile;
  onSaved: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [editing, setEditing] = useState(false);
  const [editContent, setEditContent] = useState("");
  const [saving, setSaving] = useState(false);

  const handleEdit = () => {
    setEditContent(file.content ?? "");
    setEditing(true);
  };

  const handleCancel = () => {
    setEditing(false);
    setEditContent("");
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await api.agentFileSave(agentId, file.name, editContent);
      setEditing(false);
      onSaved();
    } catch (err) {
      console.error("Failed to save file:", err);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="bg-surface-1 rounded-xl border border-surface-3/50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-4 py-3 text-left hover:bg-surface-2/50 transition-colors"
      >
        <FileText className="w-3.5 h-3.5 text-slate-500 flex-shrink-0" />
        <span className="text-xs font-medium text-slate-300">{file.name}</span>
        {file.content != null ? (
          <span className="ml-auto text-[10px] text-slate-600">
            {file.content.split("\n").length} lines
          </span>
        ) : (
          <span className="ml-auto text-[10px] text-slate-600 italic">not found</span>
        )}
      </button>

      {expanded && file.content != null && (
        <div className="border-t border-surface-3/30">
          {editing ? (
            <div className="p-3 space-y-2">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-64 bg-surface-0 text-slate-200 text-xs font-mono rounded-lg p-3 border border-surface-3/50 resize-y focus:outline-none focus:border-brand-500/50"
                spellCheck={false}
              />
              <div className="flex gap-2 justify-end">
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
          ) : (
            <div className="px-4 py-3 max-h-80 overflow-auto relative group">
              <button
                onClick={(e) => { e.stopPropagation(); handleEdit(); }}
                className="absolute top-2 right-2 inline-flex items-center gap-1 px-2 py-1 text-[10px] text-slate-400 hover:text-slate-200 bg-surface-2 rounded opacity-0 group-hover:opacity-100 transition-opacity"
              >
                <Pencil className="w-2.5 h-2.5" />
                Edit
              </button>
              <div
                className="prose-chat text-[12px]"
                dangerouslySetInnerHTML={{ __html: marked.parse(file.content) as string }}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Agent Detail View ───────────────────────────────────────────────────

function AgentDetail({
  agent,
  agentsData,
  onBack,
  onAgentUpdated,
}: {
  agent: AgentInfo;
  agentsData: AgentsData;
  onBack: () => void;
  onAgentUpdated: () => void;
}) {
  const [agentStatus, setAgentStatus] = useState<AgentStatusData | null>(null);
  const defaults = agentsData.defaults;

  useEffect(() => {
    const refresh = () => {
      api.agentStatus().then(setAgentStatus).catch(console.error);
    };
    refresh();
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, []);

  const workspaceFiles = agent.workspaceFiles ?? [];

  return (
    <div className="p-6 space-y-6 overflow-auto">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button
          onClick={onBack}
          className="p-1.5 rounded-lg hover:bg-surface-2 transition-colors text-slate-400 hover:text-slate-200"
        >
          <ChevronLeft className="w-4 h-4" />
        </button>
        {agent.avatar ? (
          <img
            src={`/mc/${agent.avatar.replace("avatars/", "")}`}
            alt={agent.id}
            className="w-8 h-8 rounded-lg object-cover"
          />
        ) : (
          <Bot className="w-5 h-5 text-brand-400" />
        )}
        <h2 className="text-lg font-semibold">{agent.name ?? agent.id}</h2>
        <span className="text-xs text-slate-500 font-mono">{agent.primaryModel}</span>
      </div>

      {/* Live Status Banner */}
      <AgentLiveStatus status={agentStatus} />

      {/* Stats row */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
        <InfoCard label="Active Sessions" value={String(agent.sessions.active)} sub={`${agent.sessions.total} total`} />
        <InfoCard label="Models" value={`${agent.enabledModels} / ${agent.modelCount}`} sub="enabled" />
        <InfoCard label="Disabled Tools" value={String(agent.disabledTools)} />
        <InfoCard label="Concurrency" value={`${agent.maxConcurrent ?? "-"} / ${agent.subagentMaxConcurrent ?? "-"}`} sub="agent / sub" />
      </div>

      {/* Subagent Runs */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Subagent Runs
        </div>
        {agentStatus ? (
          <SubagentRunsList status={agentStatus} />
        ) : (
          <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 animate-pulse">
            <div className="h-4 bg-surface-2 rounded w-24" />
          </div>
        )}
      </div>

      {/* Config details */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Configuration
        </div>
        <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 space-y-2 text-xs">
          <Row label="Primary Model" value={agent.primaryModel ?? defaults.model?.primary ?? "not set"} />
          {agent.modelFallbacks?.length > 0 && (
            <Row label="Fallback Models" value={agent.modelFallbacks.join(", ")} />
          )}
          {agent.toolsAllow && (
            <Row label="Allowed Tools" value={`${agent.toolsAllow.length} tools`} />
          )}
          <Row label="Bootstrap Max Chars" value={defaults.bootstrapMaxChars?.toLocaleString() ?? "-"} />
          <Row label="Compaction Mode" value={defaults.compaction?.mode ?? "-"} />
        </div>
      </div>

      {/* Allowed Tools */}
      <AgentToolsEditor
        agent={agent}
        allToolGroups={agentsData.allToolGroups}
        onUpdated={onAgentUpdated}
      />

      {/* Skills */}
      <AgentSkillsEditor
        agent={agent}
        allSkillNames={agentsData.allSkillNames}
        onUpdated={onAgentUpdated}
      />

      {/* Workspace files */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Workspace Files
        </div>
        <div className="space-y-2">
          {workspaceFiles.length > 0 ? (
            workspaceFiles.map((file) => (
              <WorkspaceFileEditor
                key={file.key}
                agentId={agent.id}
                file={file}
                onSaved={onAgentUpdated}
              />
            ))
          ) : (
            <div className="text-xs text-slate-500 italic py-4 text-center">
              No workspace files found
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InfoCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50">
      <div className="text-lg font-mono text-slate-200">{value}</div>
      <div className="text-[10px] text-slate-500 mt-0.5">
        {label}
        {sub && <span className="text-slate-600 ml-1">{sub}</span>}
      </div>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-slate-500">{label}</span>
      <span className="text-slate-300 font-mono">{value}</span>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────

export default function AgentsPage() {
  const [data, setData] = useState<AgentsData | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatusData | null>(null);

  const load = useCallback(() => {
    api.agents().then(setData).catch(console.error);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll agent status for live dot on cards
  useEffect(() => {
    const refresh = () => {
      api.agentStatus().then(setAgentStatus).catch(console.error);
    };
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  const agents = data?.agents ?? [];
  const selectedAgent = agents.find((a) => a.id === selected);

  if (selectedAgent && data) {
    return (
      <AgentDetail
        agent={selectedAgent}
        agentsData={data}
        onBack={() => setSelected(null)}
        onAgentUpdated={load}
      />
    );
  }

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Bot className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Agents</h2>
        <span className="text-xs text-slate-500">{agents.length} defined</span>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {agents.map((agent) => (
          <AgentCard
            key={agent.id}
            agent={agent}
            agentState={agentStatus?.mainAgent.state}
            onClick={() => setSelected(agent.id)}
          />
        ))}
      </div>
    </div>
  );
}
