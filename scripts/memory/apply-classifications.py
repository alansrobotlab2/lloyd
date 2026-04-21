#!/usr/bin/env python3
"""Promote classifier output into `_relationships.json`.

Reads `_pipeline/memory-graph/classified.jsonl` (produced by
classify-relationships.py) and updates `~/obsidian/facts/_relationships.json`
in place:

- For each classified edge, if new_type != original_type and confidence ≥ min:
  - Mark the old edge as expired (expired_at = now) — preserves history
  - Add a new edge with the typed relation + provenance=EXTRACTED_CLASSIFIER
- Edges where new_type == "mentions" are left unchanged (no-op).
- Edges below --min-confidence remain as-is (keeps original mentions edge).

Always writes a timestamped backup of the original file before modifying.

Usage:
  # preview — show what would change, write nothing
  .venvs/lloyd/bin/python scripts/memory/apply-classifications.py --dry-run

  # apply changes above the default confidence floor (0.6)
  .venvs/lloyd/bin/python scripts/memory/apply-classifications.py --apply

  # stricter floor
  .venvs/lloyd/bin/python scripts/memory/apply-classifications.py --apply --min-confidence 0.75
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FACTS_DIR = Path.home() / "obsidian" / "facts"
RELATIONSHIPS_FILE = FACTS_DIR / "_relationships.json"
DEFAULT_CLASSIFIED = Path.home() / "lloyd" / "_pipeline" / "memory-graph" / "classified.jsonl"
DEFAULT_MIN_CONF = 0.6


def _load_classified(path: Path) -> list[dict]:
    if not path.exists():
        print(f"[error] {path} not found", file=sys.stderr)
        sys.exit(1)
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _backup_relationships() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = RELATIONSHIPS_FILE.with_name(f"_relationships.{ts}.bak.json")
    shutil.copy2(RELATIONSHIPS_FILE, backup)
    return backup


def _build_active_index(edges: list[dict]) -> dict[tuple, int]:
    """(source, target, type) -> edge index for active (non-expired) edges."""
    idx = {}
    for i, e in enumerate(edges):
        if e.get("expired_at"):
            continue
        key = (e.get("source", ""), e.get("target", ""), e.get("type", ""))
        idx[key] = i
    return idx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--classified", type=Path, default=DEFAULT_CLASSIFIED)
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF)
    p.add_argument("--apply", action="store_true", help="Actually write changes")
    p.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    args = p.parse_args()

    if not args.apply and not args.dry_run:
        print("[info] no --apply specified → dry-run (no writes)")
        args.dry_run = True

    records = _load_classified(args.classified)
    print(f"[info] {len(records)} classification records loaded from {args.classified}")

    data = json.loads(RELATIONSHIPS_FILE.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    active_idx = _build_active_index(edges)
    print(f"[info] {len(edges)} total edges, {len(active_idx)} active")

    now = datetime.now(timezone.utc).isoformat()
    stats = {
        "total_records": len(records),
        "below_threshold": 0,
        "mentions_noop": 0,
        "already_typed": 0,
        "edge_missing": 0,
        "reclassified": 0,
    }
    transitions: Counter = Counter()
    new_type_counts: Counter = Counter()
    changes: list[dict] = []

    for rec in records:
        source = rec["source"]
        target = rec["target"]
        original_type = rec["original_type"]
        new_type = rec["new_type"]
        conf = float(rec.get("confidence", 0))

        # Confidence floor
        if conf < args.min_confidence:
            stats["below_threshold"] += 1
            continue

        # No-op if new_type is still mentions
        if new_type == original_type:
            if new_type == "mentions":
                stats["mentions_noop"] += 1
            else:
                stats["already_typed"] += 1
            continue

        key = (source, target, original_type)
        if key not in active_idx:
            stats["edge_missing"] += 1
            continue

        stats["reclassified"] += 1
        transitions[(original_type, new_type)] += 1
        new_type_counts[new_type] += 1
        changes.append({
            "idx": active_idx[key],
            "source": source,
            "target": target,
            "original_type": original_type,
            "new_type": new_type,
            "confidence": conf,
            "reason": rec.get("reason", ""),
            "model": rec.get("model", ""),
            "classified_at": rec.get("classified_at", now),
        })

    # Summary
    print()
    print("=" * 70)
    print("Reclassification plan:")
    for k, v in stats.items():
        print(f"  {k:<20} {v:>6}")
    print()
    print("Transitions (original → new):")
    for (o, n), c in transitions.most_common():
        print(f"  {c:>5}  {o:<14} → {n}")
    print()
    print("New-type distribution:")
    for t, c in new_type_counts.most_common():
        print(f"  {c:>5}  {t}")

    if args.dry_run:
        print()
        print("[dry-run] no changes written. Re-run with --apply to commit.")
        return 0

    if not changes:
        print("\n[info] no changes to apply")
        return 0

    backup = _backup_relationships()
    print(f"\n[info] backup written → {backup}")

    # Mutate edges in place: expire old, append new
    # Iterate over change list; each produces 1 expire + 1 new edge
    for ch in changes:
        old = edges[ch["idx"]]
        old["expired_at"] = now
        new_edge = {
            "source": ch["source"],
            "target": ch["target"],
            "type": ch["new_type"],
            "confidence": ch["confidence"],
            "provenance": "EXTRACTED_CLASSIFIER",
            "created_at": ch["classified_at"] or now,
            "expired_at": None,
            "source_doc": old.get("source_doc"),
            "reason": ch["reason"],
            "superseded_edge": {
                "type": ch["original_type"],
                "confidence": old.get("confidence"),
                "created_at": old.get("created_at"),
            },
            "classifier_model": ch["model"],
        }
        edges.append(new_edge)

    RELATIONSHIPS_FILE.write_text(
        json.dumps(data, indent=2, sort_keys=False), encoding="utf-8"
    )
    print(f"[info] wrote {RELATIONSHIPS_FILE} ({len(edges)} total edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
