/**
 * How a session gets named on screen, and how a running turn describes
 * itself. Shared by every surface that shows either: the chat history
 * list, the chat header, the dashboard's agent panel, and the Inner Voice
 * session picker.
 *
 * These live together because the fallback chain has to be identical
 * everywhere. A session that reads "Setting up TTS with cloned voice" in
 * the sidebar and `20260906_214751_iv1620` on the dashboard looks like two
 * different sessions.
 */

import type { TurnActivity } from '../api'

interface Titled {
  title?: string | null
  preview?: string | null
}

/**
 * The best available name for a session.
 *
 * Titles are written by the secondary model *after* a turn completes, so a
 * session that is mid-first-turn has none — and one whose title call
 * failed never will. `preview` (the opening of what the user typed) is the
 * honest second choice; the id is the last resort and at least always
 * identifies the row uniquely.
 */
export function sessionLabel(session: Titled, sessionId?: string | null): string {
  const title = (session.title ?? '').trim()
  if (title) return title
  const preview = (session.preview ?? '').trim()
  if (preview) return preview
  return (sessionId ?? '').trim() || 'Untitled session'
}

/** True when the label is a real name rather than an id fallback. */
export function hasSessionTitle(session: Titled): boolean {
  return Boolean((session.title ?? '').trim())
}

const ACTIVITY_TEXT: Record<TurnActivity['kind'], string> = {
  starting: 'starting',
  prefill: 'reading context',
  thinking: 'thinking',
  responding: 'writing reply',
  tool: 'running tool',
  working: 'working',
}

/**
 * One line describing what a turn is doing right now.
 *
 * A tool call reads as `Bash · supervisorctl restart` — the name alone
 * doesn't distinguish a quick file read from a four-minute build, which is
 * the whole reason an operator is looking at this panel.
 */
export function activityLabel(activity: TurnActivity | null | undefined): string {
  if (!activity) return ''
  const base = activity.label || ACTIVITY_TEXT[activity.kind] || activity.kind
  return activity.detail ? `${base} · ${activity.detail}` : base
}
