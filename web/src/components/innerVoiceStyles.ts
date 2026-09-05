import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Ban,
  Info,
  HelpCircle,
} from 'lucide-react'
import type { InnerVoiceObservationTrigger } from '../api'

export const ACTION_STYLES: Record<string, { color: string; bg: string; border: string; dot: string; label: string; Icon: typeof CheckCircle2 }> = {
  noop:                       { color: 'text-slate-400', bg: 'bg-slate-600/10',  border: 'border-slate-500/20', dot: 'bg-slate-400', label: 'noop',         Icon: CheckCircle2 },
  inject:                     { color: 'text-amber-400', bg: 'bg-amber-600/10',  border: 'border-amber-500/30', dot: 'bg-amber-400', label: 'inject',       Icon: Activity },
  cancel:                     { color: 'text-red-400',   bg: 'bg-red-600/10',    border: 'border-red-500/30',   dot: 'bg-red-400',   label: 'cancel',       Icon: XCircle },
  ambient:                    { color: 'text-blue-400',  bg: 'bg-blue-600/10',   border: 'border-blue-500/30',  dot: 'bg-blue-400',  label: 'ambient',      Icon: Activity },
  clarify:                    { color: 'text-purple-400',bg: 'bg-purple-600/10', border: 'border-purple-500/30',dot: 'bg-purple-400',label: 'clarify',      Icon: HelpCircle },
  // v3-only — never written by current code; kept for historical render
  deny_tool:                  { color: 'text-red-400',   bg: 'bg-red-700/15',    border: 'border-red-500/30',   dot: 'bg-red-400',   label: 'deny (v3)',    Icon: Ban },
  allow:                      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'allow (v3)',   Icon: CheckCircle2 },
  noop_budget_exhausted:      { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (budget)',Icon: AlertTriangle },
  noop_empty_content:         { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (empty)', Icon: Info },
  noop_no_ambient_channel:    { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (no ch)', Icon: Info },
  noop_ambient_failed:        { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (fail)',  Icon: AlertTriangle },
  noop_no_clarify_channel:    { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (no ch)', Icon: Info },
  noop_clarify_failed:        { color: 'text-amber-500', bg: 'bg-amber-600/5',   border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (fail)',  Icon: AlertTriangle },
  noop_inject_on_result:      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (late)',  Icon: Info },
  noop_cancel_on_result:      { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (late)',  Icon: Info },
  noop_clarify_on_result:     { color: 'text-slate-500', bg: 'bg-slate-600/5',   border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (late)',  Icon: Info },
  noop_cancel_with_pending_tools:        { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (mid-tool)', Icon: AlertTriangle },
  noop_pretool_after_cancel:             { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (cancelled)', Icon: Info },
  noop_inject_after_inject:              { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (inject suppressed)', Icon: AlertTriangle },
  noop_inject_on_cooldown:               { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (inject paced)', Icon: AlertTriangle },
  noop_deterministic_budget_exhausted:   { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (guard capped)', Icon: AlertTriangle },
  noop_cancel_unread_injects:            { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (injects unread)', Icon: AlertTriangle },
  noop_assistant_after_cancel:           { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (cancelled)', Icon: Info },
  noop_tool_result_after_cancel:         { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (cancelled)', Icon: Info },
  noop_result_after_cancel:              { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (cancelled)', Icon: Info },
  noop_goal_ambient_already_queued:      { color: 'text-slate-500', bg: 'bg-slate-600/5', border: 'border-slate-500/15', dot: 'bg-slate-500', label: 'noop (goal queued)', Icon: Info },
  noop_goal_ambient_failed:              { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (goal fail)', Icon: AlertTriangle },
  noop_goal_clarify_failed:              { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (goal fail)', Icon: AlertTriangle },
  noop_goal_attempts_not_persisted:      { color: 'text-amber-500', bg: 'bg-amber-600/5', border: 'border-amber-500/20', dot: 'bg-amber-500', label: 'noop (goal unbounded)', Icon: AlertTriangle },
  acknowledge_complete:       { color: 'text-emerald-400', bg: 'bg-emerald-600/10', border: 'border-emerald-500/30', dot: 'bg-emerald-400', label: 'agree: complete', Icon: CheckCircle2 },
}

export const TRIGGER_LABEL: Record<InnerVoiceObservationTrigger, string> = {
  assistant_message: 'iter end',
  tool_call:         'tool call',
  tool_result:       'tool result',
  result:            'turn end',
  pretool:           'pre-tool',
}

export function actionStyle(action: string) {
  const known = ACTION_STYLES[action]
  if (known) return known
  // An unmapped `noop_*` label is a guard downgrade the UI hasn't been
  // taught yet. Falling all the way back to plain `noop` styling hid the
  // fact that the observer WANTED to act — render it as a downgrade and
  // show the raw label so it is at least legible.
  if (action?.startsWith('noop_')) {
    return { ...ACTION_STYLES.noop_budget_exhausted, label: action.replace(/_/g, ' ') }
  }
  return ACTION_STYLES.noop
}

// IV `created_at` may be either:
//   • new format — local-naive ISO with `T` separator (matches primary)
//   • legacy format — `YYYY-MM-DD HH:MM:SS` from SQLite's CURRENT_TIMESTAMP, UTC
// Treat the legacy format as UTC by appending `Z`. Otherwise let the engine
// parse it as a local-naive ISO. Returns ms since epoch.
export function parseObservationTime(s: string): number {
  if (!s) return 0
  const isLegacyUtc = s.includes(' ') && !s.includes('T')
  return Date.parse(isLegacyUtc ? s.replace(' ', 'T') + 'Z' : s)
}
