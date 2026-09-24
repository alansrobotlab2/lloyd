/**
 * The read half of the run-to-transcript join.
 *
 * `workers/pool.py` binds `sessions_io.current_run_sessions` around each claimed
 * job and writes the collected list into the run's `meta_json`, and a
 * session-backed source stamps a singular `session_id` into the same blob. For
 * the join's whole life the pool was the only reader: the Background tab
 * received the row with the ids still inside one opaque string, so a run on the
 * Sources panel could not be followed to the transcript it produced — which is
 * the claim `architecture/background-runs.md` §7 makes about it.
 *
 * `app/routers/workers.py::run_session_ids` applies the same rule server-side
 * and ships it as a parsed `session_ids` array. This module is the mirror: the
 * array is what the endpoint sends, `meta_json` is what an older payload or a
 * hand-built row carries, and reading both keeps the Sources panel honest about
 * a run whichever one arrived.
 *
 * Deliberately free of React and of `api.ts`, so it is a plain function vitest
 * can run in `environment: "node"` — the pattern `sessionLabel.ts` established.
 * The click it feeds is asserted by a person on the running UI, because no
 * renderer is installed here to assert it in CI.
 */

/** The two fields a worker run row can carry the join in. Both optional: a row
 *  from before the endpoint parsed it has only `meta_json`. */
export interface RunSessionFields {
  session_ids?: string[] | null
  meta_json?: string | null
}

function addIds(out: string[], value: unknown): void {
  // A source that stamped one id wrote a string, not a one-element array, so
  // both shapes are legal and both mean one transcript. Whitespace-only ids are
  // treated as no id, matching the server.
  const candidates = Array.isArray(value) ? value : [value]
  for (const candidate of candidates) {
    if (typeof candidate === 'string' && candidate.trim()) {
      const id = candidate.trim()
      if (!out.includes(id)) out.push(id)
    }
  }
}

/** Parse a run's `meta_json`. A malformed blob is no ids, never a throw: the
 *  column is written by the pool and truncated rows exist, and a Sources panel
 *  that dies on one row shows no runs at all. */
function parseMeta(metaJson: unknown): unknown {
  if (typeof metaJson !== 'string' || !metaJson.trim()) return null
  try {
    return JSON.parse(metaJson)
  } catch {
    return null
  }
}

/**
 * Every transcript this run row names, in the order it recorded them.
 *
 * The endpoint's parsed array leads because the pool bound the collected list
 * first; anything the raw blob names that the array somehow lacks is appended.
 * De-duplicated, so a run whose `session_ids` and `session_id` name one session
 * gets one control, not two.
 *
 * `[]` is the normal answer, not a failure: some sources record no transcript by
 * design — `architecture/background-runs.md` §12 lists them, `automod-regression`
 * and `autoresearch` among them — and a run that dies before its first
 * `create_session` names none, so those rows render no control.
 */
export function runTranscriptIds(run: RunSessionFields | null | undefined): string[] {
  if (!run || typeof run !== 'object') return []
  const ids: string[] = []
  addIds(ids, run.session_ids)

  const meta = parseMeta(run.meta_json)
  if (meta && typeof meta === 'object' && !Array.isArray(meta)) {
    const blob = meta as Record<string, unknown>
    addIds(ids, blob.session_ids)
    addIds(ids, blob.session_id)
  }
  return ids
}

/**
 * A compact label for one transcript, for a row that has ~40ch to spend.
 *
 * The ids the recorder mints are `<yyyymmdd>_<hhmmss>_<source>_<hash>`, and the
 * leading two chunks are the same information the row's own relative time
 * already gives. The tail is the part that distinguishes two runs of one source,
 * so that is what gets rendered while `title` keeps the full id. An id that
 * does not have that shape is rendered whole rather than guessed at.
 */
export function transcriptLabel(sessionId: string): string {
  const parts = sessionId.split('_')
  return parts.length >= 3 ? parts.slice(2).join('_') : sessionId
}
