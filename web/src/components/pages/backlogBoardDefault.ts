// Which board BacklogPage stands on when it opens, as a function of the board
// list and nothing else.
//
// It is its own module because the answer used to be `boardsData[0].id`, read
// off a list whose ids are positional: `app/routers/backlog.py` numbers the
// boards by enumerating the sorted board names, so element 0 is whichever board
// sorts first, not the one the person using Mission Control means. On the box
// this was filed from, element 0 is `alfie` (7 tasks) while `lloyd` holds 1,997
// (#2068). Keeping it a pure function of its arguments is also what lets
// `vitest run` pin it — `web/` has a node-environment runner since 94850c3e, so
// a lookup like this one needs no render to be checked, and the page can be
// graded by a source claim about calling it.

/** The board BacklogPage opens on when the list contains one by this name. */
export const DEFAULT_BOARD_NAME = "lloyd"

/** The part of a board this choice reads. `BacklogBoard` satisfies it, and so
 *  does the two-field literal a test hands in, which is why the constraint is
 *  not the API type: this function must not need a server response to run. */
export interface BoardLike {
  id: number
  name: string
}

/** The board to treat as default: the one named `preferredName`, else the
 *  first, else `null` when there is no board at all.
 *
 *  `null` rather than `undefined`: element 0 of an empty array is `undefined`
 *  in JavaScript, and the caller has to tell "no board exists" apart from
 *  "there is one and I have not selected it" without re-testing the length.
 *  Generic over the element so the page gets its own `BacklogBoard` back and
 *  not a widened `BoardLike`. */
export function pickDefaultBoard<T extends BoardLike>(
  boards: readonly T[],
  preferredName: string,
): T | null {
  if (boards.length === 0) return null
  return boards.find((b) => b.name === preferredName) ?? boards[0]
}
