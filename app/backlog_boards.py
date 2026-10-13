"""The backlog's board list — the one place a board name is defined.

Every writer that creates or moves a backlog item checks the name here:
`agent_mcp/backlog.py::_handle_write` (the `backlog_write_task` tool),
`app/routers/backlog.py` (Mission Control's create and update, and the
guardian's alert filing, which POSTs to create without a board), and
`scripts/automod/backlog.py::new_item` (red-tree items and the loop's own
filings). Before the list existed a board was whatever string the writer
typed: by 2026-09-27 the vault held nineteen items across eight boards nobody
had set up (`default`, `skills`, `memory`, `Autonomy`, …), ten of them guardian
alerts that fell through the create route's old `"default"` fallback.

Adding a board is a one-line change to `BOARDS`. Reads never check the list: a
hand-edited file on another board still renders, so the board shows it rather
than hiding it — and `board_ids` below is what keeps such a file from reaching
the ids a configured board hands out.
"""

from __future__ import annotations

from collections.abc import Iterable

BOARDS: tuple[str, ...] = ("lloyd", "personal", "alfie")

# Where an item goes when its writer names no board (the guardian's alerts,
# Mission Control's create with no board picked).
DEFAULT_BOARD = "lloyd"


class UnknownBoard(ValueError):
    """A write named a board that is not in `BOARDS`."""


def board_ids(names: Iterable[str]) -> dict[str, int]:
    """{board_name: id} for every name in `names`, allocated from `BOARDS` first.

    The numeric `board_id` exists for one caller: a browser's saved board filter
    (and any other client that read `/api/backlog/boards` earlier and kept the
    numbers). Mission Control's modal and every file writer send the *name*. So
    a configured board's id is its position in `sorted(BOARDS)` and nothing
    else — `alfie=1, lloyd=2, personal=3`, whatever is on disk — and the id a
    client read yesterday still names the same board tomorrow.

    That is #2659. The ids used to be positional over every name a file
    happened to carry, so one hand-edited or restored item naming a fourth board
    (`board: Autonomy`, or none at all, which reads as `default`) inserted
    alphabetically and slid `lloyd` off the id a stale tab was holding; both
    write routes then honoured that shifted-but-valid id with a 200 and filed
    the item on the board behind it. A reader cannot reject the causing file —
    reads deliberately accept an unlisted board so its items stay visible — so
    the id had to stop depending on the corpus.

    A name outside `BOARDS` is still listed and still gets an id, allocated
    above the configured block so it can never impersonate one of them. Its own
    id is rank-based among strays, and that is safe for the one reason a
    rank-based id is ever safe: a stray cannot be *written* (`check_board`
    refuses it), so re-ranking one can at worst point a read filter at another
    stray for one refresh. Configured ids, which a write does honour, are never
    rank-based.
    """
    ids = {name: idx + 1 for idx, name in enumerate(sorted(BOARDS))}
    strays = sorted({name for name in names if name not in ids})
    for offset, name in enumerate(strays):
        ids[name] = len(BOARDS) + 1 + offset
    return ids


def check_board(name: object) -> str:
    """Return `name` stripped if it is a known board, else raise `UnknownBoard`.

    Exact match after stripping whitespace: `Lloyd` is refused rather than
    folded, because a case-folded write would still leave the caller believing
    its own spelling is a board.
    """
    board = name.strip() if isinstance(name, str) else ""
    if board not in BOARDS:
        raise UnknownBoard(f"Unknown board {name!r}. Boards: {', '.join(BOARDS)}")
    return board
