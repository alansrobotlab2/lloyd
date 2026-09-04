#!/usr/bin/env python3
"""Promote v4 classifier output into the edge store.

Reads every `classified-v4*.jsonl` under `_pipeline/memory-graph/`,
deduplicates by (source, target) keeping the most recent `classified_at`,
and lands upgrades onto active `mentions` edges via `edges.retype`.

For each record where `new_type != "mentions"` and `confidence ≥ --min-confidence`:
- Expire the matching active `(source, target, mentions)` edge
- Add a typed edge with provenance=EXTRACTED_CLASSIFIER_V4 and
  `superseded_edge_id` pointing back at the mentions edge

`retype` also expires any OTHER active edge on the same (source, target)
pair, so one pair carries one typed relation. The JSON version could leave
two active rows for a pair when the classifier had seen it twice — nine such
pairs were live on 2026-09-03.

The whole run is ONE transaction: a crash halfway through leaves the graph
untouched rather than half-upgraded. A backup is taken first anyway.

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
from app.paths import VAULT_KG_DB
from app.kg_store import KGStore

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
    """(source, target) → edge id for active `mentions` edges with an
    eligible provenance (extractor-generated only).

    Only `mentions` is indexed because v4 only re-types from there. If a
    pair already has a typed edge from a prior apply, the v4 record's
    matching `mentions` row will already be expired and won't appear here
    — that's the idempotence guard. Provenance gate ensures we don't
    overwrite human-stated, inferred, or prior-classifier-judged edges."""
    idx: dict[tuple[str, str], int] = {}
    for e in edges:
        if e.get("type") != "mentions":
            continue
        if (e.get("provenance") or "") not in ELIGIBLE_PROVENANCES:
            continue
        idx[(e.get("source", ""), e.get("target", ""))] = e["id"]
    return idx


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--classified-dir", type=Path, default=CLASSIFIED_DIR)
    p.add_argument("--pattern", default=DEFAULT_GLOB,
                   help="Glob inside classified-dir (default: classified-v4*.jsonl)")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF)
    p.add_argument("--apply", action="store_true", help="Actually write changes")
    p.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    p.add_argument("--db", type=Path, default=VAULT_KG_DB)
    args = p.parse_args()

    if not args.apply and not args.dry_run:
        print("[info] no --apply specified → dry-run (no writes)")
        args.dry_run = True

    records = _load_v4_records(args.classified_dir, args.pattern)

    st = KGStore(args.db)
    active = st.edges.active()
    active_mentions = _build_active_mentions_index(active)
    print(f"[info] {st.edges.count(active_only=False)} total edges, {len(active)} active, "
          f"{len(active_mentions)} eligible active mentions edges "
          f"(type=mentions, provenance in {sorted(ELIGIBLE_PROVENANCES)})")

    now = datetime.now(timezone.utc).isoformat()
    stats = {
        "total_records": len(records),
        "below_threshold": 0,
        "still_mentions": 0,
        "no_eligible_edge": 0,
        "duplicate_pair": 0,
        "upgrades": 0,
    }
    seen_pairs: set[tuple[str, str]] = set()
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

        def _dir_check(dc):
            """Normalize direction_check to a string verdict."""
            if isinstance(dc, dict):
                return dc.get("verdict", "")
            return dc or ""

        if key not in active_mentions:
            # Check reversed: the classifier may have classified (A, B)
            # but the live mentions edge is (B, A).
            rev = (target, source)
            if rev in active_mentions:
                new_type = rec["new_type"]
                dc_val = _dir_check(rec.get("direction_check"))
                va = rec.get("verdict_adjustment")

                if new_type == "related_to":
                    # Symmetric: just flip to match live edge
                    source, target = target, source
                    key = rev
                elif va == "downgraded_reversed":
                    # Classifier caught the flip; keep as mentions
                    stats["still_mentions"] += 1
                    continue
                elif dc_val == "reversed":
                    # Classifier caught the flip; keep as mentions
                    stats["still_mentions"] += 1
                    continue
                else:
                    # Asymmetric verb with direction_check == "correct"
                    # but live edge is reversed — the classifier's
                    # direction_check was against (source, target) as
                    # stored. Since live has (target, source), this means
                    # the direction is actually wrong for the live edge.
                    stats["still_mentions"] += 1
                    continue
            else:
                # No live edge at all
                stats["no_eligible_edge"] += 1
                continue

        if key in seen_pairs:
            # Two records for one pair after the reversed-key fold. The first
            # already consumed the edge; a second retype would expire the
            # relation it just created.
            stats["duplicate_pair"] += 1
            continue
        seen_pairs.add(key)
        stats["upgrades"] += 1
        transitions[("mentions", new_type)] += 1
        new_type_counts[new_type] += 1
        changes.append({
            "edge_id": active_mentions[key],
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
        st.close()
        return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = args.db.parent / "store-backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = st.backup(backup_dir / f"kg-v4-{ts}.sqlite")
    print(f"\n[info] backup written → {backup}")

    # One transaction for the whole run: 3,902 retypes either all land or
    # none do.
    with st.transaction():
        for ch in changes:
            st.edges.retype(
                ch["edge_id"],
                {
                    "type": ch["new_type"],
                    "confidence": ch["confidence"],
                    "provenance": "EXTRACTED_CLASSIFIER_V4",
                    "created_at": ch["classified_at"] or now,
                    "evidence": ch.get("reason_quote"),
                    "reason": ch["reason"],
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
                },
                origin="classifier",
                reason=f"v4 reclassified mentions → {ch['new_type']}",
            )
    after = st.stats()
    print(f"[info] applied {len(changes)} retypes; store now {after}")
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
