"""One definition of "a backlog task's `status`".

Five lists name this vocabulary — ``scripts/autoimplement/backlog.py`` for the
triage/implement loop, ``agent_mcp/backlog.py`` for the ``backlog_*`` tools,
``app/routers/backlog.py`` for the Mission Control board's writer,
``app/routers/dashboard.py`` for the board's counters, and ``STATUSES`` in the
React ``BacklogPage`` — and each was written by hand.

They all agree on the four words, so drift was never the bug. The bug is that
**no reader had a case for a value outside them**, and the two halves of the
system then disagreed about such an item in the worst possible direction:
``dashboard._BACKLOG_CLOSED`` counted it *open*, while
``backlog.OPEN_STATUSES`` could not see it at all. An item in that gap is
shown to a human as work in progress and is invisible to every machine that
would move it — including ``reconcile_statuses``, which forms opinions only
about items already in ``open_items`` and so is structurally unable to reach
the one thing it exists to correct.

Two items sat there since April 2026: #287 (``review``) and #304
(``closed``), stranded when the vocabulary was narrowed to these four and
nothing migrated what was already on disk.

The mapping is deliberately lopsided. Calling a word terminal when it is not
sends a live item to ``done``, where nothing will ever look at it again;
calling it non-terminal when it is sends a finished item to ``draft``, where
triage reads it, says ``already_done`` and closes it. The second costs one
triage run and is self-correcting, so only the words already known to mean
"off the board" are terminal and *everything* else becomes ``draft``.

It lives in ``app/`` and imports only the standard library for the same reason
``backlog_tags`` does: ``scripts/autoimplement/backlog.py`` is the light module
the autoimplement CLI loads, and it must not pull ``mcp`` and ``httpx`` in
behind a twenty-line helper.
"""

from __future__ import annotations

from typing import Any

# The vocabulary. Order is the pipeline's own:
#   draft ──autotriage confirms──▶ up_next ──round opens──▶ in_progress ──▶ done
PIPELINE_STATUSES: tuple[str, ...] = ("draft", "up_next", "in_progress", "done")

#: Retired spellings that meant `done`. Exactly the non-`done` members of the
#: set `dashboard._BACKLOG_CLOSED` has carried all along — that list is the
#: only record in the tree of which legacy words took a task off the board,
#: and inventing more here would be guessing at which of them are terminal.
CLOSED_ALIASES: frozenset[str] = frozenset({"closed", "cancelled", "wontfix"})

#: Every status that takes a task off the board, canonical and legacy. The
#: dashboard's counters read this.
CLOSED_STATUSES: frozenset[str] = frozenset({"done"}) | CLOSED_ALIASES

#: Statuses the loop treats as live work.
OPEN_STATUSES: frozenset[str] = frozenset(PIPELINE_STATUSES) - {"done"}


def canonical_status(value: Any) -> str:
    """Map any frontmatter `status` onto one of `PIPELINE_STATUSES`.

    Case- and whitespace-insensitive, because `Done` is a human writing the
    same word rather than a new one. An empty or missing status is `draft`,
    matching `load_item`'s long-standing default.
    """
    s = str(value or "").strip().lower()
    if not s:
        return "draft"
    if s in PIPELINE_STATUSES:
        return s
    if s in CLOSED_ALIASES:
        return "done"
    return "draft"


def is_off_vocabulary(value: Any) -> bool:
    """True when what is *on disk* is not literally one of the four.

    Deliberately case-sensitive, unlike `canonical_status`. `status: Done`
    strands exactly as `status: review` does — neither reader lowercases, so
    the dashboard counts it open and `open_items` cannot see it — and a
    rescue that tolerated it would leave that item in the gap forever.

    An *absent* status is not off-vocabulary: `load_item` and the dashboard
    both already default it to `draft`, so the two halves agree and there is
    nothing on disk to correct.
    """
    s = str(value or "").strip()
    return bool(s) and s not in PIPELINE_STATUSES
