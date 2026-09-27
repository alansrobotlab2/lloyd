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
than hiding it.
"""

from __future__ import annotations

BOARDS: tuple[str, ...] = ("lloyd", "personal", "alfie")

# Where an item goes when its writer names no board (the guardian's alerts,
# Mission Control's create with no board picked).
DEFAULT_BOARD = "lloyd"


class UnknownBoard(ValueError):
    """A write named a board that is not in `BOARDS`."""


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
