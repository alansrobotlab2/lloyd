#!/usr/bin/env python3
"""Promote v4 classifier output into `_relationships.json`.

Reads every `classified-v4*.jsonl` under `_pipeline/memory-graph/`,
deduplicates by (source, target) keeping the most recent `classified_at`,
and lands upgrades onto active `mentions` edges in the live relationships
index (resolved via `app.paths.VAULT_FACTS_ROOT`).

For each record where `new_type != "mentions"` and `confidence ≥ --min-confidence`:
- Mark the matching active `(source, target, mentions)` edge expired
- Append a new edge with the typed relation + provenance=EXTRACTED_CLASSIFIER_V4

Always writes a timestamped backup before mutating. Idempotent: re-runs
that find no new upgrades make no writes (apart from the backup, which is
suppressed when there are zero changes).

Usage:
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py            # dry-run
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py --apply
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py --apply --min-confidence 0.75
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR

RELATIONSHIPS_FILE = FACTS_DIR / "_relationships.json"
CLASSIFIED_DIR = Path(__file__).resolve().parent.parent.parent / "_pipeline" / "memory-graph"
DEFAULT_GLOB = "classified-v4*.jsonl"
DEFAULT_MIN_CONF = 0.6

# Live edges with these `provenance` values are eligible to be re-typed.
# Anything else (`STATED`, `INFERRED`, prior `EXTRACTED_CLASSIFIER*` outputs)
# represents human or prior-classifier intent we shouldn't override here.
ELIGIBLE_PROVENANCES = frozenset({"EXTRACTED"})


def _load_v4_records(classified_dir: Path, pattern: str) -> list[dict]:
    """Read every classified-v4*.jsonl, dedupe by (source, target)
    keeping the most recent `classified_at`."""
    files = sorted(classified_dir.glob(pattern))
    if not files:
        print(f"[error] no files matched {classified_dir / pattern}", file=sys.stderr)
        sys.exit(1)
    print(f"[info] reading {len(files)} file(s):")
    for f in files:
        print(f"        {f}")

    by_pair: dict[tuple[str, str], dict] = {}
    raw_lines = 0
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                raw_lines += 1
                s = r.get("source")
                t = r.get("target")
                if not s or not t:
                    continue
                key = (s, t)
                prev = by_pair.get(key)
                if prev is None or _classified_at(r) > _classified_at(prev):
                    by_pair[key] = r
    print(f"[info] {raw_lines} raw lines → {len(by_pair)} unique pairs (after dedup)")
    return list(by_pair.values())


def _classified_at(rec: dict) -> str:
    """Sortable timestamp for tie-breaking. Empty string sorts before any ISO."""
    return str(rec.get("classified_at") or "")


def _build_active_mentions_index(edges: list[dict]) -> dict[tuple[str, str], int]:
    """(source, target) → edge index for active `mentions` edges with an
    eligible provenance (extractor-generated only).

    Only `mentions` is indexed because v4 only re-types from there. If a
    pair already has a typed edge from a prior apply, the v4 record's
    matching `mentions` row will already be expired and won't appear here
    — that's the idempotence guard. Provenance gate ensures we don't
    overwrite human-stated, inferred, or prior-classifier-judged edges."""
    idx: dict[tuple[str, str], int] = {}
    for i, e in enumerate(edges):
        if e.get("expired_at"):
            continue
        if e.get("type") != "mentions":
            continue
        if (e.get("provenance") or "") not in ELIGIBLE_PROVENANCES:
            continue
        idx[(e.get("source", ""), e.get("target", ""))] = i
    return idx


def _backup_relationships() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = RELATIONSHIPS_FILE.with_name(f"_relationships.{ts}.v4.bak.json")
    shutil.copy2(RELATIONSHIPS_FILE, backup)
    return backup


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--classified-dir", type=Path, default=CLASSIFIED_DIR)
    p.add_argument("--pattern", default=DEFAULT_GLOB,
                   help="Glob inside classified-dir (default: classified-v4*.jsonl)")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF)
    p.add_argument("--apply", action="store_true", help="Actually write changes")
    p.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    args = p.parse_args()

    if not args.apply and not args.dry_run:
        print("[info] no --apply specified → dry-run (no writes)")
        args.dry_run = True

    records = _load_v4_records(args.classified_dir, args.pattern)

    data = json.loads(RELATIONSHIPS_FILE.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    active_mentions = _build_active_mentions_index(edges)
    print(f"[info] {len(edges)} total edges, "
          f"{len(active_mentions)} eligible active mentions edges "
          f"(type=mentions, provenance in {sorted(ELIGIBLE_PROVENANCES)})")

    now = datetime.now(timezone.utc).isoformat()
    stats = {
        "total_records": len(records),
        "below_threshold": 0,
        "still_mentions": 0,
        "no_eligible_edge": 0,
        "upgrades": 0,
    }
    transitions: Counter = Counter()
    new_type_counts: Counter = Counter()
    changes: list[dict] = []

    for rec in records:
        new_type = rec.get("new_type")
        if not new_type:
            continue

        # We only upgrade away from `mentions`. Direction-flip downgrades
        # like `uses → mentions` (verdict_adjustment=downgraded_reversed)
        # already match the existing edge type and produce no change.
        if new_type == "mentions":
            stats["still_mentions"] += 1
            continue

        conf = float(rec.get("confidence") or 0)
        if conf < args.min_confidence:
            stats["below_threshold"] += 1
            continue

        source = rec["source"]
        target = rec["target"]
        key = (source, target)
        if key not in active_mentions:
            # Either no live edge at all, the live edge is no longer
            # `mentions` (e.g., upgraded by a prior apply), or the live
            # edge has a non-EXTRACTED provenance we deliberately leave
            # alone.
            stats["no_eligible_edge"] += 1
            continue

        stats["upgrades"] += 1
        transitions[("mentions", new_type)] += 1
        new_type_counts[new_type] += 1
        changes.append({
            "idx": active_mentions[key],
            "source": source,
            "target": target,
            "new_type": new_type,
            "confidence": conf,
            "reason": rec.get("reason", ""),
            "model": rec.get("model", ""),
            "classified_at": rec.get("classified_at") or now,
            "verdict_adjustment": rec.get("verdict_adjustment"),
            "direction_check": rec.get("direction_check"),
            "quote_verified": rec.get("quote_verified"),
            "reason_quote": rec.get("reason_quote"),
            "src_type_hint": rec.get("src_type_hint"),
            "tgt_type_hint": rec.get("tgt_type_hint"),
            "prompt_version": rec.get("prompt_version", "v4"),
        })

    print()
    print("=" * 70)
    print("v4 reclassification plan:")
    for k, v in stats.items():
        print(f"  {k:<28} {v:>6}")
    print()
    print("Transitions (mentions → new_type):")
    for (o, n), c in transitions.most_common():
        print(f"  {c:>5}  {o:<10} → {n}")
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

    for ch in changes:
        old = edges[ch["idx"]]
        old["expired_at"] = now
        new_edge = {
            "source": ch["source"],
            "target": ch["target"],
            "type": ch["new_type"],
            "confidence": ch["confidence"],
            "provenance": "EXTRACTED_CLASSIFIER_V4",
            "created_at": ch["classified_at"] or now,
            "expired_at": None,
            "source_doc": old.get("source_doc"),
            "reason": ch["reason"],
            "superseded_edge": {
                "type": "mentions",
                "confidence": old.get("confidence"),
                "created_at": old.get("created_at"),
            },
            "classifier_model": ch["model"],
            "classifier_meta": {
                "prompt_version": ch["prompt_version"],
                "verdict_adjustment": ch["verdict_adjustment"],
                "direction_check": ch["direction_check"],
                "quote_verified": ch["quote_verified"],
                "reason_quote": ch["reason_quote"],
                "src_type_hint": ch["src_type_hint"],
                "tgt_type_hint": ch["tgt_type_hint"],
            },
        }
        edges.append(new_edge)

    RELATIONSHIPS_FILE.write_text(
        json.dumps(data, indent=2, sort_keys=False), encoding="utf-8"
    )
    print(f"[info] wrote {RELATIONSHIPS_FILE} ({len(edges)} total edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
