#!/usr/bin/env python3
"""
Content Hasher — SHA256-based change detection for nightly processing.

Maintains a hash index at ~/lloyd-data/_pipeline/content-hashes.json.
Before processing a file, check if its hash has changed since last run.
Unchanged files can be skipped entirely.

Usage:
    from content_hasher import ContentHasher

    hasher = ContentHasher()
    changed_files = hasher.get_changed_files(list_of_paths)
    # ... process only changed_files ...
    # Pass the digest of the bytes you read whenever you have one; a bare path
    # is re-read here, and a re-read after processing records content nobody
    # processed as processed (see update_hash).
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
from app.paths import production_data_root  # noqa: E402


# LLOYD_CONTENT_HASHES lets a rebuild keep its own index. Without it the
# rebuild would skip every file the LIVE tree had already extracted and
# produce an empty tree. The default is the live data root, as the nightly's
# lock that guards it is (nightly_extraction._STATE_PIPELINE), unless
# LLOYD_DATA explicitly names another.
DEFAULT_INDEX_PATH = Path(os.environ["LLOYD_CONTENT_HASHES"]) \
    if os.environ.get("LLOYD_CONTENT_HASHES") \
    else Path(os.environ.get("LLOYD_DATA") or production_data_root()) / "_pipeline" / "content-hashes.json"


def count_extracted(index_data: dict) -> int:
    """How many documents an index payload says were read to the end.

    #1151 gave an entry a second meaning: besides the digest, it records how far
    of the file the pass that wrote it got. So `file_count` — every entry,
    including one whose document was cut off at the chunk budget — is no longer a
    count of documents that were extracted. A reader that needs that number must
    ask the entries, not the scalar: a gate that promotes a corpus built from
    unread tails has exactly the defect the coverage records were added to catch.

    An entry counts unless its own record says the tail is owed. A record with no
    `complete` key predates those records and says nothing either way; treating
    one as not-extracted would rewrite the meaning of every index written before
    #1151, so `nightly_extraction` refutes such a record by document size when it
    can and leaves it alone when it cannot.
    """
    return sum(1 for entry in (index_data.get("hashes") or {}).values()
               if not (isinstance(entry, dict) and entry.get("complete") is False))


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
        """Does this file still need processing?

        Two ways to need it, not one. The obvious one: the bytes moved under the
        recorded digest. The other — the one this method could not express until
        #1151 — the recorded digest is a perfect match and still stands for less
        than the file, because the pass that recorded it stopped at the end of its
        chunk budget with the tail unread. An entry recorded that way carries
        `complete: false`, and reading its digest as "done" is how a 957,489-char
        feed sat in this index at 4.9% coverage, re-skipped every night.

        An entry written before #1151 carries no `complete` key and therefore
        keeps the old answer: digest match means done. Defaulting that the other way
        would put every legacy entry back in the queue at once — however many the
        index holds the morning this ships, and the count moves with pruning —
        turning one night's extraction into a re-extraction of the corpus. The
        bounded migration lives in `nightly_extraction._extract_all_facts`, which
        re-reads a digest-less entry only when the file is too long for a single
        pass to have finished it.

        The entry keeps its digest in that case; what the flag changes is only
        this answer. Withholding the digest instead would restart the next pass at
        offset 0 forever, and a 955k-char document would never be finished either.
        """
        key = str(path)
        current_hash = self._hash_file(path)
        if not current_hash:
            return True  # Can't read = treat as changed
        stored = self._hashes.get(key, {})
        if isinstance(stored, str):
            return stored != current_hash  # legacy format: bare hash string
        if stored.get("sha256") != current_hash:
            return True
        return not stored.get("complete", True)

    def coverage(self, path: Path) -> tuple:
        """`(chars_covered, sha256)` for one path, from the stored entry.

        `chars_covered` is -1 when the entry records no coverage — every entry
        written before #1151, and any written by a caller that hashed a file it
        never read — which is not the same as 0: it asserts nothing about how much
        of the document was read, so a caller cannot treat it as finished. An
        unknown path is `(0, "")`.
        """
        stored = self._hashes.get(str(path), {})
        if isinstance(stored, str):
            return (-1, stored)
        covered = stored.get("covered_through")
        return (-1 if covered is None else int(covered), stored.get("sha256", ""))

    def get_changed_files(self, paths: list[Path]) -> list[Path]:
        """Filter a list of paths to only those that have changed."""
        return [p for p in paths if self.has_changed(p)]

    def update_hash(self, path: Path, sha256: Optional[str] = None,
                    covered_through=None, complete: bool = True):
        """Record the digest of one file, and how far of it that pass read.

        `covered_through` is the char offset the recording pass stopped at and
        `complete` says whether that offset was the end of the document. Both
        default to what a caller that only knows the digest can mean — "the whole
        file, done" — so every caller that predates #1151 keeps its old meaning and
        no legacy entry becomes unfinished just by being read by new code. The
        nightly extraction pass is the one caller that reads a bounded window and
        must therefore say so: it records `complete=False` with the offset it
        reached, and `has_changed` answers "changed" from that flag even while the
        digest still matches.

        Pass `sha256` — the digest of the bytes the caller actually read — from
        any code path that read the file itself. Re-hashing here instead records
        whatever the file looks like at flush time, and a flush can be the end
        of a run that takes 545-1666s (tests/test_extraction_single_instance.py:11-13),
        so a note appended to during the run would be stored as extracted at
        content nobody read and the appended lines would never be fact-extracted
        (#482). This is the gate half of the hazard documented at
        `app/atomic_io.py::hash_bytes`; the provenance half was fixed there, the
        gate half was not.

        With no digest — or an empty one, which is what a caller that read
        nothing offers — the file on disk is hashed. That is the CLI `--update`
        route at the bottom of this file, which hashes files it never extracted
        and has nothing to carry.
        """
        key = str(path)
        digest = (sha256 or "").strip() or self._hash_file(path)
        if digest:
            self._hashes[key] = {
                "sha256": digest,
                "checked_at": datetime.now(timezone.utc).isoformat(),
                # `None` covered_through is every caller that has no coverage to
                # report — the CLI route below, and every entry that predates
                # #1151. `coverage()` reads that as -1 ("nothing is asserted"),
                # never as "0 chars read".
                "covered_through": (None if covered_through is None
                                    else max(0, int(covered_through))),
                "complete": True if covered_through is None else bool(complete),
            }

    def update_hashes(self, paths):
        """Record digests for several files.

        An item is a bare path (hashed from disk), a `(path, sha256)` pair — the
        nightly extraction checkpoint, which still holds the digest of the bytes it
        extracted — or `(path, sha256, covered_through, complete)` once the caller
        knows how far of the document it read (#1151).
        """
        for item in paths:
            if isinstance(item, tuple):
                self.update_hash(*item)
            else:
                self.update_hash(item)

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
