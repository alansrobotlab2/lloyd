"""The mitigation drill's latest result per surface, on disk (#703).

`scripts/mitigation_drill.py` measures what each stop control actually stops;
this file keeps the most recent measurement of each so `GET /api/workers/status`
can say which controls are in-flight stops, which are dispatch-only, and how
long the last stop took — without running anything. A surface is merged in, not
replaced wholesale, so a drill that measured one surface keeps the other's
last reading.

The reader never raises: a status route is most useful when something is wrong,
so a missing or unreadable file is reported as a never-run marker.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app import paths
from app.atomic_io import atomic_write_text

NEVER_RUN = {"state": "never-run"}


def _path(path: Optional[Path]) -> Path:
    # Read at call time, so a test that points `paths.MITIGATION_DRILL_STATE`
    # elsewhere is honoured.
    return Path(path) if path is not None else paths.MITIGATION_DRILL_STATE


def record(surfaces: list[dict], *, path: Optional[Path] = None,
           at: Optional[str] = None) -> dict[str, Any]:
    """Merge each drilled surface's result into the state file, atomically."""
    target = _path(path)
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        merged = dict(json.loads(target.read_text(encoding="utf-8")).get("surfaces") or {})
    except Exception:
        merged = {}  # absent or unreadable: this drill's results start it again
    for s in surfaces:
        merged[s["surface"]] = {"classification": s.get("classification"),
                                "seconds": s.get("seconds"),
                                "ok": s.get("ok"), "at": at}
    state = {"surfaces": merged}
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def read(*, path: Optional[Path] = None) -> dict[str, Any]:
    """`{surface: {classification, seconds, at}}`, or `NEVER_RUN` (with an
    `error` when the file exists but cannot be read). Never raises."""
    try:
        surfaces = json.loads(_path(path).read_text(encoding="utf-8")).get("surfaces")
        out = {str(name): {"classification": r.get("classification"),
                           "seconds": r.get("seconds"), "at": r.get("at")}
               for name, r in (surfaces or {}).items() if isinstance(r, dict)}
        return out or dict(NEVER_RUN)
    except FileNotFoundError:
        return dict(NEVER_RUN)
    except Exception as e:  # unreadable is reported, never raised
        return {**NEVER_RUN, "error": f"{type(e).__name__}: {e}"}
