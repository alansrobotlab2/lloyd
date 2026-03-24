/**
 * Shared tool layout definitions for Tools page and Agent tools editor.
 * Groups tools by source, then by function, with icons and colors.
 */
import {
  Brain, Database, CheckSquare, GitFork, Clock,
  FolderOpen, Globe, Sparkles, Terminal, Mail, Mic,
  Cpu, Wrench, Search
} from "lucide-react";

export type FuncGroup = {
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  tools: string[];
};

export type SourceMeta = {
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  label: string;
};

// ── Functional sub-groups per source ─────────────────────────────────────

export const OPENCLAW_FUNC_GROUPS: FuncGroup[] = [
  { label: "Sessions & Agents", icon: GitFork, color: "text-cyan-400", tools: ["sessions_list", "sessions_history", "sessions_send", "sessions_spawn", "subagents", "session_status", "agents_list", "message"] },
  { label: "Files & Runtime", icon: FolderOpen, color: "text-sky-400", tools: ["read", "write", "edit", "apply_patch", "exec", "process"] },
  { label: "Web & Memory", icon: Globe, color: "text-violet-400", tools: ["web_search", "web_fetch", "memory_search", "memory_get"] },
  { label: "System & Media", icon: Clock, color: "text-orange-400", tools: ["cron", "gateway", "nodes", "browser", "canvas", "image", "tts"] },
];

export const MCP_FUNC_GROUPS: FuncGroup[] = [
  { label: "Memory (Next-Gen)", icon: Brain, color: "text-violet-400", tools: ["get_facts", "add_fact", "get_relations", "add_relation", "context_bundle", "get_profile", "detect_contradictions", "resolve_contradictions", "rebuild_index"] },
  { label: "Memory (Vault)", icon: Database, color: "text-indigo-400", tools: ["mem_search", "mem_get", "mem_write", "tag_search", "tag_explore", "vault_overview", "prefill_context", "work_mode"] },
  { label: "Backlog", icon: CheckSquare, color: "text-emerald-400", tools: ["backlog_boards", "backlog_tasks", "backlog_get_task", "backlog_write_task"] },
  { label: "Files", icon: FolderOpen, color: "text-sky-400", tools: ["file_read", "file_write", "file_edit", "file_patch", "file_glob", "file_grep"] },
  { label: "Shell", icon: Terminal, color: "text-green-400", tools: ["run_bash", "bg_exec", "bg_process"] },
  { label: "Web", icon: Globe, color: "text-blue-400", tools: ["http_search", "http_fetch", "http_request"] },
  { label: "Skills", icon: Sparkles, color: "text-amber-400", tools: ["skills_search", "skills_get"] },
];

export const VOICE_FUNC_GROUPS: FuncGroup[] = [
  { label: "Voice", icon: Mic, color: "text-amber-400", tools: ["voice_last_utterance", "voice_enroll_speaker", "voice_list_speakers"] },
];

export const THUNDERBIRD_FUNC_GROUPS: FuncGroup[] = [
  { label: "Email & Calendar", icon: Mail, color: "text-rose-400", tools: ["email_accounts", "email_folders", "email_search", "email_read", "email_recent", "calendar_list", "calendar_events", "contacts_search", "contacts_get"] },
];

// ── Source metadata ──────────────────────────────────────────────────────

export const SOURCE_META: Record<string, SourceMeta> = {
  "openclaw": { icon: Cpu, color: "text-slate-400", label: "OpenClaw (Built-in)" },
  "mcp-tools": { icon: Terminal, color: "text-indigo-400", label: "MCP Tools" },
  "voice-tools": { icon: Mic, color: "text-amber-400", label: "Voice Tools" },
  "thunderbird-tools": { icon: Mail, color: "text-rose-400", label: "Thunderbird" },
};

// ── Helpers ──────────────────────────────────────────────────────────────

export function getFuncGroups(source: string): FuncGroup[] | null {
  if (source.startsWith("openclaw")) return OPENCLAW_FUNC_GROUPS;
  if (source.startsWith("mcp-tools")) return MCP_FUNC_GROUPS;
  if (source.startsWith("voice")) return VOICE_FUNC_GROUPS;
  if (source.startsWith("thunderbird")) return THUNDERBIRD_FUNC_GROUPS;
  return null;
}

export function getSourceMeta(source: string): SourceMeta {
  for (const [prefix, meta] of Object.entries(SOURCE_META)) {
    if (source.startsWith(prefix)) return meta;
  }
  return { icon: Wrench, color: "text-slate-400", label: source };
}

export function getSourceKey(source: string): string {
  for (const prefix of Object.keys(SOURCE_META)) {
    if (source.startsWith(prefix)) return prefix;
  }
  return source;
}

export const UNGROUPED_ICON = Search;
