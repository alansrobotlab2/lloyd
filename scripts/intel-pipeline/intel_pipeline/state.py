"""State store helpers for tracking seen items."""

import json
import os
from pathlib import Path
from typing import Optional

from ._paths import STATE_FILE, RAW_DIR, SCAN_SERIES_FILE


def ensure_dirs():
    """Ensure required directories exist."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    """Load scanner state from JSON file."""
    ensure_dirs()
    if not STATE_FILE.exists():
        return {"seen": set(), "last_run": None}
    
    with open(STATE_FILE, "r") as f:
        data = json.load(f)
    
    # Convert set to list for JSON serialization, back to set
    if "seen" in data and isinstance(data["seen"], list):
        data["seen"] = set(data["seen"])
    if "seen" not in data:
        data["seen"] = set()
    
    return data


def save_state(state: dict):
    """Save scanner state to JSON file."""
    ensure_dirs()
    
    # Convert set to list for JSON serialization
    data = state.copy()
    if isinstance(data.get("seen"), set):
        data["seen"] = list(data["seen"])
    
    with open(STATE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def is_seen(item_id: str, state: Optional[dict] = None) -> bool:
    """Check if an item ID has been seen."""
    if state is None:
        state = load_state()
    return item_id in state.get("seen", set())


def mark_seen(item_id: str, state: Optional[dict] = None):
    """Mark an item ID as seen."""
    if state is None:
        state = load_state()
    
    if "seen" not in state:
        state["seen"] = set()
    state["seen"].add(item_id)
    save_state(state)


def get_raw_path(date_str: str) -> Path:
    """Get path for raw items file for a given date."""
    ensure_dirs()
    return RAW_DIR / f"{date_str}.jsonl"


def save_raw_items(items: list, date_str: str):
    """Save raw items to JSONL file for a date."""
    ensure_dirs()
    raw_path = get_raw_path(date_str)
    
    with open(raw_path, "a") as f:
        for item in items:
            if hasattr(item, "to_json"):
                f.write(item.to_json() + "\n")
            else:
                f.write(json.dumps(item) + "\n")


def load_raw_items(date_str: str) -> list:
    """Load raw items from JSONL file for a date."""
    raw_path = get_raw_path(date_str)
    if not raw_path.exists():
        return []
    
    from .models import FeedItem
    
    items = []
    with open(raw_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Convert dict to FeedItem
                item = FeedItem.from_dict(data)
                items.append(item)
    return items


def append_scan_series(row: dict) -> "Path":
    """Append one per-scan row to the coverage/delivery series (#3873) and return its path.

    Append-only and never rewritten, which is the whole point: the numbers this records are
    the ones no other artefact keeps. `scanner-state.json` answers "what did the LAST scan
    reach" and `raw/<date>.jsonl` answers "how many rows exist for this day"; only a series
    answers "when did the YouTube half stop delivering", and a run that exits 1 on a
    degraded stage still books its row before it is withheld its success banner — a blackout
    is exactly the run whose row matters.

    The path is read from the module global at CALL time, like `get_raw_path` reads
    `RAW_DIR`, so a test can point the series at a temporary vault; a failure to write it
    raises, because a silent recorder is how a series stops existing.
    """
    ensure_dirs()
    path = Path(SCAN_SERIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def count_raw_rows(date_str: str, source: "str | None" = None) -> int:
    """How many rows `raw/<date>.jsonl` holds, optionally of one `source` (#3873).

    Counted from the FILE rather than from a scanner's return value, because the number the
    blackout is measured by is the one a human can reproduce with `wc -l` on the day file —
    which is how #3873 measured itself. A missing day file counts as zero, which is what a
    scan that appended nothing produced.
    """
    path = get_raw_path(date_str)
    try:
        with open(path) as f:
            lines = [ln for ln in f if ln.strip()]
    except FileNotFoundError:
        return 0
    if source is None:
        return len(lines)
    n = 0
    for ln in lines:
        try:
            row = json.loads(ln)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("source") == source:
            n += 1
    return n
