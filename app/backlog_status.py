"""One definition of "a backlog task's `status`".

Five lists name this vocabulary — ``scripts/automod/backlog.py`` for the
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
``backlog_tags`` does: ``scripts/automod/backlog.py`` is the light module
the automod CLI loads, and it must not pull ``mcp`` and ``httpx`` in
behind a twenty-line helper.
"""

from __future__ import annotations

import re
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

#: The words a caller reaches for when it means "not done yet" and has not read
#: the enum. Not a synonym list and not up for growth: these are the exact three
#: spellings that appear in the run records — 10 rejected
#: `backlog_write_task(status=…)` calls in 9 run records across 9 tasks (24, 60,
#: 67, 76 ×2, 78, 80, 83, 84, 92), `open` ×6, `todo` ×3, `backlog` ×1, measured
#: 2026-11-24 over the seven-day window #4525 was filed from — by anchoring on the
#: tool payload:
#:
#:   find ~/lloyd-data/autonomy-runs -name 'run_*.md' -newermt '-7 days' -print0 \
#:     | xargs -0 grep -o 'success": false, "error": "Invalid status' | wc -l
#:
#: In the same corpus `Invalid board`, `Invalid priority` and
#: `Invalid description_mode` occurred zero times, and all three of those fields
#: declare an `enum` in the very schema `status` did not — which is the actual
#: defect, and the enum is the fix. This set is only what an honest caller means
#: by the word: `closed` is a retired *terminal* word (see `CLOSED_ALIASES`), so
#: mapping it from a write would be the one mistake this module refuses to make.
STATUS_GUESS_WORDS: frozenset[str] = frozenset({"open", "todo", "backlog"})

#: Where a guessed word lands. Never `up_next`: `app/autonomy.py` dispatches only
#: from `up_next`, and a write of it is ruled on by the promotion ledger rather
#: than applied — so aliasing to it would have changed the error's shape instead
#: of removing it, and could arm an untriaged item. `draft` is the state
#: `canonical_status` already sends every unknown word to, for the reason in this
#: module's own docstring: a wrong `draft` costs one triage run and self-corrects.
GUESS_TARGET: str = "draft"

#: Statuses a guessed word must not move an item *off*. `up_next → draft` is how
#: task #68 went dark for ~50 cycles (`app/autonomy.py`, the comment at `:3393`),
#: and a demotion of a closed item buries a finished task back on the board.
_PROTECTED_STATUSES: frozenset[str] = frozenset({"up_next", "in_progress", "done"})


def _guess_word(value: Any) -> str:
    """The bare word a status value spells: trimmed, lower-cased, delimiters gone.

    Deliberately narrower than `canonical_status`: that one maps *anything* it
    does not recognise, while this only normalises the spelling of a value that is
    then tested against `STATUS_GUESS_WORDS`. `"Open"`, `" TODO "` and `"Backlog"`
    are the same guesses a hand-written call produces, and `canonical_status`
    already treats `Done` and `done` as one word, so refusing these two for
    padding would be a stricter rule than the vocabulary itself.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def invalid_status_message(value: Any) -> str:
    """The refusal every write surface answers an unknown word with.

    Byte-identical to what the two surfaces said before #4525, and pinned as a
    literal by `tests/test_backlog_status_vocabulary.py`, because a caller that
    recovers from this error is recovering by reading the four legal words out of
    it — that sentence is the interface.
    """
    return (f"Invalid status '{value}'. Must be one of: "
            + ", ".join(sorted(PIPELINE_STATUSES)))


def resolve_status(value: Any, current: Any = None) -> tuple[str, dict | None]:
    """Resolve a requested status against the status the item carries now.

    Returns `(accepted_status, coercion_or_None)`, and raises `ValueError` with
    `invalid_status_message(value)` for anything it may not write. One function,
    called by all three writers — `agent_mcp/backlog.py`'s `backlog_write_task`,
    `POST /api/backlog/task-update` and `POST /api/backlog/task-create` — because
    #518 already taught this tree what happens when only one of the two backlog
    writers is fixed.

    The two rules, and why each is shaped that way:

    * A word in `STATUS_GUESS_WORDS` is written as `GUESS_TARGET`, and only when
      `current` is absent or already `draft` — which includes a create, whose own
      default is `draft`. Anywhere else it is the refusal, because a guess must
      not be able to move a live item backwards.
    * Any other word outside `PIPELINE_STATUSES` keeps today's refusal. Coercing
      the words nobody was observed guessing would be inventing vocabulary, and
      `tests/test_backlog_status_vocabulary.py` refuses that on both surfaces.
    """
    if value in PIPELINE_STATUSES:
        return value, None
    # `canonical_status`, not `_guess_word`, for the guard: the protected words are
    # `up_next` and `in_progress`, and `_guess_word` exists to fold human spelling
    # *including its delimiters* — it would read `"up_next"` as `"upnext"`, match
    # nothing, and let a guess demote the armed pool. Canonically, the current
    # status is either `draft` (including an absent one, and an off-vocabulary
    # stranded word, which every reader already treats as untriaged) or it is not.
    if (_guess_word(value) in STATUS_GUESS_WORDS
            and canonical_status(current) not in _PROTECTED_STATUSES):
        return GUESS_TARGET, {
            "from": value,
            "to": GUESS_TARGET,
            "why": (f"'{value}' is not a backlog status. Written as '{GUESS_TARGET}' — "
                    f"the state of an item nobody has triaged yet — because a wrong "
                    f"'{GUESS_TARGET}' costs one triage run and self-corrects, while "
                    f"'up_next' is the armed pool this loop dispatches from. Legal "
                    f"values: {', '.join(sorted(PIPELINE_STATUSES))}."),
        }
    raise ValueError(invalid_status_message(value))


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
