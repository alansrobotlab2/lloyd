import { describe, expect, it } from 'vitest'
import { DEFAULT_BOARD_NAME, pickDefaultBoard } from './backlogBoardDefault'

// Boards in the shape the route returns them and in the order the route returns
// them: `app/routers/backlog.py` enumerates the sorted board names, so ids are
// positional and element 0 is `alfie`. That order is the whole of #2068 — the
// page's mount default was element 0, which on that box is a 7-task board whose
// tasks are all `status: done` and therefore outside #1213's seven-day window,
// while the 1,997-task `lloyd` board sat at index 1.
const BOARDS = [
  { id: 1, name: 'alfie' },
  { id: 2, name: 'lloyd' },
  { id: 3, name: 'personal' },
]

describe('pickDefaultBoard', () => {
  it('takes the preferred board by name even when another sorts first', () => {
    expect(pickDefaultBoard(BOARDS, 'lloyd')).toEqual({ id: 2, name: 'lloyd' })
  })

  it('falls back to the first board when no name matches', () => {
    // A board list with no `lloyd` in it — a fresh checkout, or a renamed
    // board — still has to land somewhere, and "the first" is what the page
    // did before and still does in that case.
    expect(pickDefaultBoard(BOARDS, 'nonexistent')).toEqual({ id: 1, name: 'alfie' })
  })

  it('returns null for an empty list rather than an undefined element', () => {
    expect(pickDefaultBoard([], 'lloyd')).toBe(null)
  })

  it('follows the name across a renumbering instead of keeping the index', () => {
    // A board created before `lloyd` alphabetically renumbers everything after
    // it, so the id this returns has to be read off the list it was given.
    const withNew = [{ id: 1, name: 'aardvark' }, ...BOARDS.map((b) => ({ ...b, id: b.id + 1 }))]
    expect(pickDefaultBoard(withNew, 'lloyd')).toEqual({ id: 3, name: 'lloyd' })
  })

  it('is a lookup, not a filter-and-take-first: an exact name wins over order', () => {
    const named = [{ id: 9, name: 'zulu' }, { id: 4, name: 'lloyd' }]
    expect(pickDefaultBoard(named, 'lloyd')?.id).toBe(4)
    expect(pickDefaultBoard(named, 'zulu')?.id).toBe(9)
  })

  it('asks for the board the page means, spelled the way the route spells it', () => {
    // The name is the seam: `app/routers/backlog.py` keys boards by directory
    // name, so a string that differs by case or spacing matches nothing here
    // and the page silently goes back to the positional default.
    expect(DEFAULT_BOARD_NAME).toBe('lloyd')
    expect(pickDefaultBoard(BOARDS, DEFAULT_BOARD_NAME)?.id).toBe(2)
  })
})
