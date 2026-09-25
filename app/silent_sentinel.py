"""The one reading of the `[SILENT]` sentinel (#1507).

An autonomy run with nothing to report ends on exactly `[SILENT]` and nothing
else; that declines delivery to the user. Four sites used to read it two ways
— the run record and the Discord helper by exact match on the whole text, the
scheduled-task adapter and the `/api/autonomy/health` rollup by substring — so
a run that merely *mentioned* the token in its report was recorded as a real
run, had its notification dropped, and was counted as a declined run in the
health rate. Exact match is canonical: it is what the run record writes, and
it is the contract the prompt states ("exactly `[SILENT]` (nothing else)").

Stdlib only, so the Discord helper and the worker adapter can import it
without pulling in `autonomy`.
"""

from __future__ import annotations

from typing import Any, Mapping

SILENT_SENTINEL = "[SILENT]"


def is_silent_response(text: str | None) -> bool:
    """True when `text` IS the sentinel — the whole of it, whitespace aside.

    Callers hand it the run's terminal block; a text that contains the token
    among other words is a report, not a decline.
    """
    return (text or "").strip() == SILENT_SENTINEL


def run_is_silent(meta: Mapping[str, Any] | None, text: str | None) -> bool:
    """A recorded run's verdict: its own `meta.silent` when it carries one.

    The run record judges the terminal block, which no downstream text slice
    can reproduce once a run has narrated before signing off, so the flag wins
    whenever it is present. Rows that predate the flag fall back to the same
    exact predicate on whatever text they have.
    """
    if meta and "silent" in meta:
        return bool(meta.get("silent"))
    return is_silent_response(text)
