import { useEffect, useState, useCallback } from "react";
import { Bot, ChevronLeft, ChevronDown, ChevronRight, Cpu, Wrench, Sparkles, Users, Layers, FileText, Pencil, X, Save } from "lucide-react";
import { marked } from "marked";
import { api, AgentInfo, AgentsData, AgentStatusData, SubagentRunInfo, ToolGroupInfo, WorkspaceFile, CallLogEntry, SdkAgentInfo, SdkAgentsData, CcInstanceInfo, CcInstanceMessage } from "../../api";

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
            src={agent.avatar}
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
        {agent.identity && (
          <div className="text-xs text-slate-400 italic text-right flex-shrink-0 max-w-[45%] leading-snug">
            {agent.identity}
          </div>
        )}
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

// ── Agent Call Log ──────────────────────────────────────────────────────

function formatRelTs(ts: string): string {
  const diffMs = Date.now() - new Date(ts).getTime();
  if (diffMs < 1000) return "just now";
  const s = Math.floor(diffMs / 1000);
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  return `${Math.floor(m / 60)}h ago`;
}

function formatArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args);
  if (keys.length === 0) return "";
  const first = keys[0];
  const val = String(args[first] ?? "").slice(0, 40);
  return keys.length === 1 ? val : `${val} +${keys.length - 1}`;
}

function AgentCallLog({ agentId }: { agentId: string }) {
  const [entries, setEntries] = useState<CallLogEntry[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetch = () => {
      api.agentCallLog(agentId, 30)
        .then((d) => { if (!cancelled) setEntries(d.entries); })
        .catch(() => { if (!cancelled) setEntries([]); });
    };
    fetch();
    const interval = setInterval(fetch, 3000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [agentId]);

  if (entries === null) {
    return (
      <div className="bg-surface-1 rounded-xl p-4 border border-surface-3/50 animate-pulse">
        <div className="h-4 bg-surface-2 rounded w-32" />
      </div>
    );
  }

  if (entries.length === 0) {
    return (
      <div className="text-xs text-slate-500 italic py-4 text-center">No activity yet</div>
    );
  }

  return (
    <div className="bg-surface-1 rounded-xl p-3 border border-surface-3/50 space-y-0">
      {[...entries].reverse().map((entry, i) => (
        <div key={i} className="flex items-start gap-2.5 py-1.5 border-b border-surface-3/15 last:border-0">
          {entry.type === "llm" ? (
            <>
              <div className="mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-indigo-400/70" />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-[10px] font-mono bg-surface-0 px-1.5 py-0.5 rounded border border-surface-3/50 text-indigo-300">
                    {entry.model ?? "llm"}
                  </span>
                  {entry.inputTokens !== undefined && (
                    <span className="text-[10px] text-slate-500 font-mono">
                      {entry.inputTokens}→{entry.outputTokens}tok
                    </span>
                  )}
                  {entry.cost !== undefined && (
                    <span className="text-[10px] text-slate-600 font-mono">
                      ${entry.cost.toFixed(4)}
                    </span>
                  )}
                </div>
              </div>
            </>
          ) : (
            <>
              <div className={`mt-1.5 w-1.5 h-1.5 rounded-full flex-shrink-0 ${entry.isError ? "bg-red-400" : "bg-emerald-400/70"}`} />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <span className="text-[10px] font-mono bg-surface-0 px-1.5 py-0.5 rounded border border-surface-3/50 text-brand-300">
                    {entry.toolName}
                  </span>
                  {formatArgs(entry.args ?? {}) && (
                    <span className="text-[10px] text-slate-500 font-mono truncate max-w-[160px]">
                      {formatArgs(entry.args ?? {})}
                    </span>
                  )}
                </div>
                {entry.resultPreview && (
                  <div className="text-[10px] text-slate-600 mt-0.5 truncate">{entry.resultPreview}</div>
                )}
              </div>
            </>
          )}
          <span className="text-[10px] text-slate-600 font-mono flex-shrink-0">{formatRelTs(entry.ts)}</span>
        </div>
      ))}
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
            src={agent.avatar}
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

      {/* Call Log */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">
          Call Log
        </div>
        <AgentCallLog agentId={agent.id} />
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

// ── SDK Agent Card ──────────────────────────────────────────────────────

// ── Coffee Mug SVG ──────────────────────────────────────────────────────

const MUG_COLORS: Record<string, string> = {
  coder: "#e67e22",
  researcher: "#3498db",
  reviewer: "#9b59b6",
  tester: "#27ae60",
  planner: "#f39c12",
  auditor: "#e74c3c",
  operator: "#1abc9c",
  orchestrator: "#2c3e50",
  clawhub: "#e91e63",
};

function CoffeeMug({ agentId, isWorking }: { agentId: string; isWorking: boolean }) {
  const color = MUG_COLORS[agentId] || "#7f8c8d";
  return (
    <svg width="28" height="28" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg" className="flex-shrink-0">
      {/* Smoke lines — only visible when working */}
      {isWorking && (
        <g>
          <path className="mug-smoke-line" d="M10 12 Q10 9 11 7" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0" />
          <path className="mug-smoke-line" d="M14 11 Q13.5 8 15 6" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0" />
          <path className="mug-smoke-line" d="M18 12 Q18.5 9 17.5 7" stroke="#94a3b8" strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0" />
        </g>
      )}
      {/* Mug body */}
      <rect x="6" y="13" width="16" height="14" rx="2" fill={color} />
      {/* Coffee surface */}
      <rect x="7" y="14" width="14" height="3" rx="1" fill="#3e2723" opacity="0.5" />
      {/* Handle */}
      <path d="M22 16 Q27 16 27 21 Q27 26 22 26" stroke={color} strokeWidth="2.5" fill="none" strokeLinecap="round" />
    </svg>
  );
}

const MODEL_COLORS: Record<string, string> = {
  opus: "text-purple-400",
  sonnet: "text-blue-400",
  haiku: "text-emerald-400",
};

function SdkAgentCard({
  agent,
  instanceCounts,
  ccInstances,
}: {
  agent: SdkAgentInfo;
  instanceCounts: Record<string, { active: number; recent: number }>;
  ccInstances: CcInstanceInfo[];
}) {
  const counts = instanceCounts[agent.id] || { active: 0, recent: 0 };
  const [expanded, setExpanded] = useState(false);
  const agentInstances = ccInstances.filter((i) =>
    (i.type === "orchestrate" && agent.id === "orchestrator") ||
    (i.type === "spawn" && i.agent === agent.id)
  );

  return (
    <div className="bg-surface-1 rounded-xl border border-surface-3/50 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 text-left hover:bg-surface-2/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <img
            src={agent.avatarUrl}
            alt={agent.id}
            className="w-10 h-10 rounded-xl object-cover flex-shrink-0"
            onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-slate-200">{agent.id}</span>
              <span className={`text-[10px] font-mono ${MODEL_COLORS[agent.model] || "text-slate-400"}`}>
                {agent.model}
              </span>
              {counts.active > 0 && (
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.5)]" />
                  <span className="text-[10px] text-cyan-400 font-mono">{counts.active} running</span>
                </div>
              )}
            </div>
            <div className="text-[10px] text-slate-500 mt-0.5 line-clamp-1">{agent.description}</div>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="text-right">
              <div className="text-[10px] text-slate-500 font-mono">{agent.tools.length} tools</div>
              {agent.hasMcp && <div className="text-[10px] text-slate-600 font-mono">+MCP</div>}
            </div>
            {expanded ? <ChevronDown className="w-4 h-4 text-slate-500" /> : <ChevronRight className="w-4 h-4 text-slate-500" />}
          </div>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-surface-3/30 pt-3">
          {/* Tools */}
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">Tools</div>
            <div className="flex flex-wrap gap-1">
              {agent.tools.map((t) => (
                <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-surface-2 text-slate-400 font-mono">{t}</span>
              ))}
              {agent.mcpTools.map((t) => (
                <span key={t} className="text-[10px] px-1.5 py-0.5 rounded bg-brand-600/20 text-brand-400 font-mono">
                  {t.replace("mcp__openclaw-tools__", "")}
                </span>
              ))}
            </div>
          </div>

          {/* Config */}
          <div className="flex gap-4 text-[10px] text-slate-500 font-mono">
            <span>maxTurns: {agent.maxTurns}</span>
            <span>model: {agent.model}</span>
          </div>

          {/* Recent instances */}
          {agentInstances.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-1">
                Recent Instances ({agentInstances.length})
              </div>
              <div className="space-y-1.5">
                {agentInstances.slice(0, 5).map((inst) => (
                  <div key={inst.id} className="flex items-center gap-2 text-[10px]">
                    <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${
                      inst.status === "running" ? "bg-cyan-400" :
                      inst.status === "complete" ? "bg-emerald-400" :
                      inst.status === "error" ? "bg-red-400" : "bg-slate-500"
                    }`} />
                    <span className="text-slate-400 truncate flex-1">{inst.task}</span>
                    <span className="text-slate-600 font-mono flex-shrink-0">
                      {inst.turns}t · ${inst.costUsd.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── CC Instance Card ─────────────────────────────────────────────────────

const CC_STATUS_COLORS: Record<string, string> = {
  running: "bg-cyan-400 shadow-[0_0_6px_rgba(34,211,238,0.5)]",
  complete: "bg-emerald-400",
  error: "bg-red-400",
  aborted: "bg-slate-500",
};

const CC_STATUS_BADGES: Record<string, string> = {
  running: "bg-cyan-500/20 text-cyan-400 border-cyan-500/30",
  complete: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
  error: "bg-red-500/20 text-red-400 border-red-500/30",
  aborted: "bg-slate-500/20 text-slate-400 border-slate-500/30",
};

function CcInstanceCard({ instance, onClick }: { instance: CcInstanceInfo; onClick: () => void }) {
  const elapsed = formatElapsed(instance.elapsedMs);
  const taskPreview = instance.task.length > 80 ? instance.task.slice(0, 80) + "\u2026" : instance.task;
  const statusColor = CC_STATUS_COLORS[instance.status] ?? CC_STATUS_COLORS.aborted;
  const statusBadge = CC_STATUS_BADGES[instance.status] ?? CC_STATUS_BADGES.aborted;

  return (
    <button
      onClick={onClick}
      className="w-full text-left bg-surface-1 rounded-xl p-4 border border-surface-3/50 hover:border-cyan-500/30 hover:bg-surface-2/30 transition-all"
    >
      <div className="flex items-start gap-3">
        <div className="mt-1 flex-shrink-0 relative">
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          {instance.status === "running" && (
            <div className="absolute inset-0 w-2 h-2 rounded-full bg-cyan-400 animate-ping opacity-40" />
          )}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono border ${statusBadge}`}>
              {instance.status}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-2 text-slate-400 font-mono border border-surface-3/50">
              {instance.type}
            </span>
            {instance.agent && (
              <span className="text-[10px] text-slate-500 font-mono">{instance.agent}</span>
            )}
          </div>
          <div className="text-xs text-slate-300 mt-1.5 line-clamp-2">{taskPreview}</div>
          {instance.activity && instance.status === "running" && (
            <div className="text-[10px] text-cyan-400 mt-1 truncate font-mono">{instance.activity}</div>
          )}
        </div>
        <div className="text-right flex-shrink-0 text-[10px] text-slate-500 font-mono space-y-0.5">
          <div>{elapsed}</div>
          <div>${instance.costUsd.toFixed(3)}</div>
          <div>{instance.turns}t</div>
        </div>
      </div>
    </button>
  );
}

// ── CC Instance Log Panel ────────────────────────────────────────────────

function CcInstanceLogPanel({ instance, onBack }: { instance: CcInstanceInfo; onBack: () => void }) {
  const [messages, setMessages] = useState<CcInstanceMessage[]>([]);
  const [loading, setLoading] = useState(true);

  const loadLog = useCallback(() => {
    api.ccInstanceLog(instance.id, 200).then((d) => {
      setMessages(d.messages);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, [instance.id]);

  useEffect(() => {
    loadLog();
  }, [loadLog]);

  // Auto-refresh while running
  useEffect(() => {
    if (instance.status !== "running") return;
    const interval = setInterval(loadLog, 3000);
    return () => clearInterval(interval);
  }, [instance.status, loadLog]);

  function msgTypeColor(type: string): string {
    switch (type) {
      case "tool_use": return "text-blue-400";
      case "subagent_start": return "text-cyan-400";
      case "subagent_end": return "text-emerald-400";
      case "error": return "text-red-400";
      case "task_progress": return "text-amber-400";
      default: return "text-slate-300";
    }
  }

  function msgTypeLabel(type: string): string {
    switch (type) {
      case "tool_use": return "tool";
      case "subagent_start": return "spawn";
      case "subagent_end": return "done";
      case "task_progress": return "task";
      case "error": return "error";
      default: return "text";
    }
  }

  const statusBadge = CC_STATUS_BADGES[instance.status] ?? CC_STATUS_BADGES.aborted;

  return (
    <div className="p-6 space-y-4 overflow-auto">
      <div className="flex items-center gap-3">
        <button onClick={onBack} className="text-slate-400 hover:text-slate-200 transition-colors">
          <ChevronLeft className="w-5 h-5" />
        </button>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono border ${statusBadge}`}>
              {instance.status}
            </span>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-2 text-slate-400 font-mono border border-surface-3/50">
              {instance.type}
            </span>
            {instance.agent && <span className="text-[10px] text-slate-500 font-mono">{instance.agent}</span>}
          </div>
          <div className="text-sm text-slate-200 mt-1 line-clamp-2">{instance.task}</div>
        </div>
        <div className="text-right text-[10px] text-slate-500 font-mono flex-shrink-0">
          <div>{formatElapsed(instance.elapsedMs)}</div>
          <div>${instance.costUsd.toFixed(3)}</div>
          <div>{instance.turns} turns</div>
        </div>
      </div>

      <div className="bg-surface-1 rounded-xl border border-surface-3/50 overflow-hidden">
        <div className="px-4 py-2 border-b border-surface-3/30 flex items-center gap-2">
          <FileText className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-[10px] uppercase tracking-wider text-slate-500 font-medium">Work Log</span>
          <span className="text-[10px] text-slate-600 font-mono ml-auto">{messages.length} events</span>
          {instance.status === "running" && (
            <div className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
          )}
        </div>
        <div className="divide-y divide-surface-3/20 max-h-[60vh] overflow-y-auto">
          {loading ? (
            <div className="px-4 py-8 text-center text-xs text-slate-500 animate-pulse">Loading log...</div>
          ) : messages.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-slate-500">No events yet</div>
          ) : (
            messages.map((msg, i) => (
              <div key={i} className="px-4 py-2 flex gap-3 hover:bg-surface-2/30">
                <span className={`text-[10px] font-mono w-10 flex-shrink-0 mt-0.5 ${msgTypeColor(msg.type)}`}>
                  {msgTypeLabel(msg.type)}
                </span>
                <div className="flex-1 min-w-0">
                  {msg.agent && (
                    <span className="text-[10px] text-cyan-500 font-mono mr-2">[{msg.agent}]</span>
                  )}
                  <span className={`text-xs break-words whitespace-pre-wrap ${msgTypeColor(msg.type)}`}>
                    {msg.content}
                  </span>
                </div>
                <span className="text-[10px] text-slate-600 font-mono flex-shrink-0">
                  {new Date(msg.ts).toLocaleTimeString()}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Page ───────────────────────────────────────────────────────────

export default function AgentsPage({
  initialAgentId,
  onAgentIdConsumed,
}: {
  initialAgentId?: string | null;
  onAgentIdConsumed?: () => void;
} = {}) {
  const [data, setData] = useState<AgentsData | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [agentStatus, setAgentStatus] = useState<AgentStatusData | null>(null);
  const [sdkData, setSdkData] = useState<SdkAgentsData | null>(null);
  const [ccInstances, setCcInstances] = useState<CcInstanceInfo[]>([]);
  const [selectedCcInstance, setSelectedCcInstance] = useState<CcInstanceInfo | null>(null);

  const load = useCallback(() => {
    api.agents().then(setData).catch(console.error);
    api.ccAgents().then(setSdkData).catch(() => setSdkData(null));
    api.ccInstances().then((d) => setCcInstances(d.instances)).catch(() => setCcInstances([]));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Pre-select agent when navigated from another page (e.g. activity panel click)
  useEffect(() => {
    if (!initialAgentId || !data) return;
    const match = data.agents.find((a) => a.id === initialAgentId);
    if (match) setSelected(match.id);
    onAgentIdConsumed?.();
  }, [initialAgentId, data, onAgentIdConsumed]);

  // Poll agent status + CC instances
  useEffect(() => {
    const refresh = () => {
      api.agentStatus().then(setAgentStatus).catch(console.error);
      api.ccInstances().then((d) => setCcInstances(d.instances)).catch(() => {});
      api.ccAgents().then(setSdkData).catch(() => {});
    };
    refresh();
    const interval = setInterval(refresh, 5000);
    return () => clearInterval(interval);
  }, []);

  const agents = data?.agents ?? [];
  const sdkAgents = sdkData?.agents ?? [];
  const instanceCounts = sdkData?.instanceCounts ?? {};
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

  if (selectedCcInstance) {
    const freshInst = ccInstances.find(i => i.id === selectedCcInstance.id) ?? selectedCcInstance;
    return (
      <CcInstanceLogPanel instance={freshInst} onBack={() => setSelectedCcInstance(null)} />
    );
  }

  return (
    <div className="p-6 space-y-6 overflow-auto">
      <div className="flex items-center gap-3">
        <Bot className="w-5 h-5 text-brand-400" />
        <h2 className="text-lg font-semibold">Agents</h2>
        <span className="text-xs text-slate-500">
          {agents.length} core · {sdkAgents.length} SDK · {ccInstances.length} cc
        </span>
      </div>

      {/* Core agents (from openclaw.json) */}
      <div>
        <div className="text-[10px] uppercase tracking-wider text-slate-500 font-medium mb-2">
          Core Agents
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

      {/* SDK agents (from agent-orchestrator) */}
      {sdkAgents.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-cyan-400 font-medium mb-2 flex items-center gap-1.5">
            <Cpu className="w-3 h-3" />
            SDK Agents
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {sdkAgents.map((agent) => (
              <SdkAgentCard
                key={agent.id}
                agent={agent}
                instanceCounts={instanceCounts}
                ccInstances={ccInstances}
              />
            ))}
          </div>
        </div>
      )}

      {/* CC Instances */}
      {ccInstances.length > 0 && (
        <div>
          <div className="text-[10px] uppercase tracking-wider text-cyan-400/70 font-medium mb-2 flex items-center gap-1.5">
            <Layers className="w-3 h-3" />
            CC Instances
            <span className="text-slate-600 font-mono">({ccInstances.filter(i => i.status === "running").length} running)</span>
          </div>
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
            {ccInstances
              .sort((a, b) => {
                if (a.status === "running" && b.status !== "running") return -1;
                if (b.status === "running" && a.status !== "running") return 1;
                return b.startedAt - a.startedAt;
              })
              .slice(0, 20)
              .map((inst) => (
                <CcInstanceCard
                  key={inst.id}
                  instance={inst}
                  onClick={() => setSelectedCcInstance(inst)}
                />
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
