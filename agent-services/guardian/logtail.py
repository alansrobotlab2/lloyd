"""Incremental log reading by (device, inode, offset). Stdlib only.

Deliberately not `app/supervisor_client.py::_read_log_tail`, which does
`f.readlines()[-50:]` — that reads the whole file into memory, and
`logs/server.err` rotates at 10 MB, so polling it that way would mean a
multi-megabyte read every tick.

Rotation is by rename (verified: `server.err`, `server.err.1`, `server.err.2`
all have different inodes), so the cursor must key on the inode, not the path.
When the inode changes we first drain the tail of the file we *were* reading —
now `.1` — before starting the new one, otherwise every rotation silently
swallows however much had accumulated since the last tick.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class LogCursor:
    """Tracks read position across rotation and truncation."""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self._state: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            self._state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._state = {}

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def reset_to_end(self, path: str) -> None:
        """Start tracking `path` from its current end (used at promotion)."""
        try:
            st = os.stat(path)
        except OSError:
            self._state[path] = {"dev": 0, "inode": 0, "offset": 0}
            return
        self._state[path] = {"dev": st.st_dev, "inode": st.st_ino, "offset": st.st_size}

    def position(self, path: str) -> dict:
        return dict(self._state.get(path) or {"dev": 0, "inode": 0, "offset": 0})

    def read_new(self, path: str, cap_bytes: int) -> tuple[str, bool]:
        """Return (new_text, overflowed) since the last read of `path`."""
        try:
            st = os.stat(path)
        except OSError:
            return "", False

        prev = self._state.get(path)
        chunks: list[str] = []
        overflow = False

        if prev is None:
            # First sight: start at the end, do not replay history.
            self._state[path] = {"dev": st.st_dev, "inode": st.st_ino, "offset": st.st_size}
            return "", False

        rotated = (prev.get("inode") != st.st_ino) or (prev.get("dev") != st.st_dev)

        if rotated:
            # Drain whatever landed in the predecessor after our last read.
            rotated_path = f"{path}.1"
            try:
                rst = os.stat(rotated_path)
                if rst.st_ino == prev.get("inode"):
                    text, over = _read_range(rotated_path, int(prev.get("offset", 0)),
                                             rst.st_size, cap_bytes)
                    chunks.append(text)
                    overflow = overflow or over
            except OSError:
                pass
            start = 0
        elif st.st_size < int(prev.get("offset", 0)):
            # Truncated in place (`> server.err`); start over.
            start = 0
        else:
            start = int(prev.get("offset", 0))

        text, over = _read_range(path, start, st.st_size, cap_bytes)
        chunks.append(text)
        overflow = overflow or over

        self._state[path] = {"dev": st.st_dev, "inode": st.st_ino, "offset": st.st_size}
        return "".join(chunks), overflow


def _read_range(path: str, start: int, end: int, cap_bytes: int) -> tuple[str, bool]:
    if end <= start:
        return "", False
    overflow = False
    if end - start > cap_bytes:
        # More than the cap in one tick is itself a symptom, and we must not
        # let a log storm exhaust memory.
        start = end - cap_bytes
        overflow = True
    try:
        with open(path, "rb") as f:
            f.seek(start)
            data = f.read(end - start)
    except OSError:
        return "", overflow
    return data.decode("utf-8", "replace"), overflow


def bootstrap_chronic(paths: list[str], *, max_bytes: int,
                      min_distinct_hours: int) -> set[str]:
    """Signatures that recur across ≥ N distinct hours — never worth reverting for.

    One bounded backward scan, run once. This is what stops the detector from
    firing on the errors that were already there before any promotion.
    """
    import detect

    hours_by_sig: dict[str, set[str]] = {}
    for path in paths:
        for candidate in (path, f"{path}.1"):
            try:
                st = os.stat(candidate)
            except OSError:
                continue
            start = max(0, st.st_size - max_bytes)
            text, _ = _read_range(candidate, start, st.st_size, max_bytes)
            for line in text.splitlines():
                parsed = detect.parse_log_line(line)
                if not parsed or parsed["level"] not in ("ERROR", "CRITICAL"):
                    continue
                stamp = line[:13]  # "YYYY-MM-DD HH"
                hours_by_sig.setdefault(parsed["signature"], set()).add(stamp)
            for ev in detect.extract_events(text):
                if ev["level"] == "TRACEBACK":
                    hours_by_sig.setdefault(ev["signature"], set()).add("tb")
    return {sig for sig, hours in hours_by_sig.items() if len(hours) >= min_distinct_hours}
