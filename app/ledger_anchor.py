"""The one owner of the question: can a ledger row ever cover this index line?

`lloyd/MEMORY.md` and `lloyd/USER.md` are an INDEX and a LEDGER: a loaded line is
retired by a row in a companion file, keyed on the line's ANCHOR — the first
`ANCHOR_CHARS` characters of the line with its `- [type] (date)` prefix and its
emphasis markers normalised away. `anchor_of` derives it and `representable` answers,
from the anchor alone, whether a row could ever carry it. Two shapes cannot:

  * an anchor containing `|`, the row's own field delimiter (`FIELD_SEP`). The row
    parser cuts the anchor field at the first pipe, so such an anchor can never round
    trip. Any single pipe anywhere in the 60 characters does it, which is why a hook
    naming a shell pipeline (`git show X | cat`) or an alternation is unretirable even
    though it reads like every other line in the file.
  * an anchor that opens AND closes on a backtick. `_anchor_field` in
    `scripts/memory/memory_ledger.py` strips one wrapping pair to keep pre-#1730 rows
    covering their line, so this shape is indistinguishable from the legacy wrapped form.

A line of either shape is permanent load: it sits in the prompt every session and
nothing can retire it. `representable` lived inside `scripts/memory/memory_ledger.py`
while its only caller was that file's own `status` command — a reporter that is also
the only caller of its own guard cannot register failure, and 2026-11-02 shipped
`(2026-11-02) **`git show X | python3 - <<'PY'` reads no bytes…**, which `status` has
been counting as `unrepresentable: 1` ever while every writer waved it through. It
moved here so the three readers share ONE predicate:

  * `scripts/memory/memory_ledger.py` imports it for `status` (the report),
  * `app/memory_ceiling.py` imports it for `memory_write_error`, which refuses an
    unretirable line on the way in, and
  * `scripts/util/ledger_strand_findings.py` reaches it through the same module at both
    vault commit doors, so the run that landed the line sees the refusal itself.

It could not live on either side. `memory_ledger.py` already does `from
app.memory_ceiling import index_bounds_apply, tight_limit`, so importing `memory_ledger`
back from `app/memory_ceiling.py` at module scope is a cycle — the same constraint that
makes `MEMORY_LEDGERS` a deliberate duplicate in `app/memory_ceiling.py`. And importing
`app` would pull `data_root` and `prompt_surface` down with it, so this file depends on
the standard library alone.
"""

from __future__ import annotations

import re

__all__ = ["ANCHOR_CHARS", "FIELD_SEP", "norm", "anchor_of", "representable",
           "loaded_entries"]


#: How much of an entry identifies it. Long enough that two entries in one
#: file never share it (checked: 0 collisions in USER.md and MEMORY.md on
#: 2026-09-25), short enough that an edit to the end of a line keeps its row.
ANCHOR_CHARS = 60


#: What ends the anchor field. A pipe, not a backtick: the anchor is the loaded
#: line's own first 60 characters, code spans and all, and every later field in
#: the row is already pipe-separated (#1730 — see the module docstring).
FIELD_SEP = "|"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def anchor_of(entry_line: str) -> str:
    """The anchor for one loaded entry line."""
    body = entry_line.strip()
    body = re.sub(r"^(?:[-*]|>)\s*", "", body)
    return norm(body)[:ANCHOR_CHARS].rstrip()


def representable(anchor: str) -> bool:
    """Whether a row can carry `anchor` and `ledger_rows` read it back whole.

    Two shapes cannot: one containing the field delimiter, which the row parser
    cuts at, and one that both opens and closes on a backtick, which no reader
    can tell from the legacy wrapped form. A line with either is not curator
    backlog — no row will ever cover it — so `status` counts it separately.
    """
    return FIELD_SEP not in anchor and not (anchor.startswith("`")
                                            and anchor.endswith("`"))


def loaded_entries(text: str) -> list[str]:
    """Bullet lines outside the front matter: the entries a ledger row covers.

    Blockquote paragraphs (USER.md's header) are the file's own rules about
    itself and carry no row."""
    out, in_fm = [], False
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.strip() == "---":
            in_fm = True
            continue
        if in_fm:
            in_fm = line.strip() != "---"
            continue
        if line.lstrip().startswith("- "):
            out.append(line)
    return out
