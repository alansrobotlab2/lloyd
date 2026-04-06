"""State store helpers for tracking seen items."""

import json
import os
from pathlib import Path
from typing import Optional

# Default paths
STATE_FILE = Path.home() / "obsidian/memory/feeds/scanner-state.json"
RAW_DIR = Path.home() / "obsidian/memory/feeds/raw"


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
