#!/usr/bin/env python3
"""
Content Hasher — SHA256-based change detection for nightly processing.

Maintains a hash index at ~/lloyd/_pipeline/content-hashes.json.
Before processing a file, check if its hash has changed since last run.
Unchanged files can be skipped entirely.

Usage:
    from content_hasher import ContentHasher

    hasher = ContentHasher()
    changed_files = hasher.get_changed_files(list_of_paths)
    # ... process only changed_files ...
    hasher.update_hashes(changed_files)
    hasher.save()
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# LLOYD_CONTENT_HASHES lets a rebuild keep its own index. Without it the
# rebuild would skip every file the LIVE tree had already extracted and
# produce an empty tree.
DEFAULT_INDEX_PATH = Path(os.environ["LLOYD_CONTENT_HASHES"]) \
    if os.environ.get("LLOYD_CONTENT_HASHES") \
    else Path.home() / "lloyd" / "_pipeline" / "content-hashes.json"


class ContentHasher:
    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self._hashes: dict[str, dict] = {}
        self._load()

    def _load(self):
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._hashes = data.get("hashes", {})
            except Exception:
                self._hashes = {}

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(self._hashes),
            "hashes": self._hashes,
        }
        # Atomic: this is the resume point for a multi-hour extraction, and a
        # truncated index reads as "nothing has been extracted".
        from app.atomic_io import atomic_write_text
        atomic_write_text(self.index_path, json.dumps(data, indent=2))

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        try:
            h.update(path.read_bytes())
        except Exception:
            return ""
        return h.hexdigest()

    def has_changed(self, path: Path) -> bool:
        """Check if a file has changed since last recorded hash."""
        key = str(path)
        current_hash = self._hash_file(path)
        if not current_hash:
            return True  # Can't read = treat as changed
        stored = self._hashes.get(key, {})
        if isinstance(stored, str):
            return stored != current_hash  # legacy format: bare hash string
        return stored.get("sha256") != current_hash

    def get_changed_files(self, paths: list[Path]) -> list[Path]:
        """Filter a list of paths to only those that have changed."""
        return [p for p in paths if self.has_changed(p)]

    def update_hash(self, path: Path):
        """Record current hash for a file."""
        key = str(path)
        current_hash = self._hash_file(path)
        if current_hash:
            self._hashes[key] = {
                "sha256": current_hash,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }

    def update_hashes(self, paths: list[Path]):
        """Record current hashes for multiple files."""
        for p in paths:
            self.update_hash(p)

    def stats(self) -> dict:
        return {
            "tracked_files": len(self._hashes),
            "index_path": str(self.index_path),
        }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Check which files have changed since last run")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")
    parser.add_argument("--update", action="store_true", help="Update hashes after reporting")
    args = parser.parse_args()

    scan_dir = Path(args.directory).expanduser()
    if not scan_dir.is_dir():
        print(f"Not a directory: {scan_dir}", file=sys.stderr)
        sys.exit(1)

    hasher = ContentHasher()
    all_files = sorted(scan_dir.rglob(args.pattern))
    changed = hasher.get_changed_files(all_files)

    print(f"Scanned: {len(all_files)} files")
    print(f"Changed: {len(changed)} files")
    print(f"Unchanged: {len(all_files) - len(changed)} files")

    if changed:
        print("\nChanged files:")
        for f in changed[:50]:
            print(f"  {f}")
        if len(changed) > 50:
            print(f"  ... and {len(changed) - 50} more")

    if args.update:
        hasher.update_hashes(all_files)  # Update ALL files, not just changed
        hasher.save()
        print(f"\nHashes updated and saved to {hasher.index_path}")
