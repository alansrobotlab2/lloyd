"""The shapes a hand-typed size takes in a test (#3904).

Shared by the two files that legislate about one frozen corpus:
`tests/test_memory_ledger_bound.py`, whose corpus node used to pin its witness by
`len(raw)` and a newline count compared to typed literals, and
`tests/test_vault_commit_witness_warning.py`, whose rung advises the writer who has to
re-freeze it.

The rule both files enforce is the same one #3904 landed: a size that describes a mutable
file may be MEASURED off it at the moment of the assert and printed from there, but it may
not be TYPED, because a typed figure is a receipt someone has to remember at the next
freeze. #3282 recorded a freeze that carried a byte count copied out of an item's prose
instead of the new file's, and #3719 closed `done` on a literal a later freeze left stale.

A shape is a `(label, compiled pattern)` pair; a label names what the hit would break, so a
failure message can say which kind of figure reappeared rather than only where.
"""
from __future__ import annotations

import re

TYPED_SIZE_SHAPES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a length compared to a typed literal",
     re.compile(r"len\(\s*\w+\s*\)\s*==\s*[\d_]{2,}")),
    ("a newline count compared to a typed literal",
     re.compile(r"""count\(\s*(?:"\\n"|'\\n'|chr\(10\))\s*\)\s*==\s*[\d_]{2,}""")),
    ("a size figure written into prose",
     re.compile(r"\d[\d,]*(?:_\d+)*\s+(?:B\b|bytes\b|newlines?\b|lines?\b)")),
    ("a quoted `wc -c` answer", re.compile(r"`wc -c` answered \d")),
)


def typed_size_statements(src: str) -> list[tuple[str, re.Match[str]]]:
    """Every hand-typed size statement in `src`, as `(shape, match)` pairs.

    Empty means the source measures whatever it reports rather than stating it. Callers
    pass the text whose property is in question — one test node's source, or a whole
    script — so a legitimate byte figure outside that scope (the topic ceiling, a ledger
    bound) is not swept up with it.
    """
    hits: list[tuple[str, re.Match[str]]] = []
    for shape, pattern in TYPED_SIZE_SHAPES:
        found = pattern.search(src)
        if found:
            hits.append((shape, found))
    return hits


def named_typed_sizes(hits: list[tuple[str, re.Match[str]]]) -> str:
    """The scanner's findings as one clause, so every guard reports a seed identically."""
    return "; ".join(f"{shape}: {found.group()!r}" for shape, found in hits)
