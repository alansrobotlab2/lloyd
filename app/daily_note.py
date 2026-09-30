"""The backend's reach into the watchdog's daily-note header builder (#1887).

`memory/<date>.md` is written by three routes, and the front-matter block a fresh one
gets now comes from one place: `agent-services/guardian/daily_note.py`. That module is
the canonical copy rather than something under `app/` because the watchdog executes a
staged copy of `agent-services/guardian/*.py` (`lloyd-guardian.service`, via
`agent-services/bin/guardian-stage.sh`) and must be able to alert with the repo broken
— so the builder is stdlib-only, and the backend is the side that reaches across.

This module is that single crossing. Both `app/post_capture._append_daily_note` and
`app/autonomy.append_daily_alert_line` call through it, which is what keeps the three
writers byte-identical; `tests/test_daily_note_shared_header.py` asserts these names
ARE the watchdog's functions, so a rendering added here — a third copy of the literal,
which is the defect #1887 removed — fails that node rather than quietly diverging.

`agent-services` is not a package (hyphen in the directory name), so the load is by
path: `spec_from_file_location`, the same route `scripts/automod/promote.py` uses for
`agent-services/guardian/guardian.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_GUARDIAN_DIR = Path(__file__).resolve().parent.parent / "agent-services" / "guardian"
_BUILDER_PATH = _GUARDIAN_DIR / "daily_note.py"

_spec = importlib.util.spec_from_file_location("lloyd_guardian_daily_note", _BUILDER_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - a missing file is louder than this
    raise ImportError(f"cannot load the daily-note builder from {_BUILDER_PATH}")
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

fresh_front_matter = _module.fresh_front_matter
fresh_header = _module.fresh_header

__all__ = ["fresh_front_matter", "fresh_header"]

# Both names are the watchdog's own functions, bound rather than wrapped: a wrapper
# here would be a second place to keep the header, which is what #1887 is about. A
# caller that writes the whole file in one go concatenates `fresh_header(now, day)`
# with the body it appended before — the `## Sessions` section for capture, the alert
# line for the fleet watchdog.
