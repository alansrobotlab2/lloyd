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

That sweep is provenance-blind, so the pass — not `retype` — is where the
"never re-type human or inferred intent" guard rail is enforced (#1453). A pair
carrying an active edge whose provenance is neither the extractor's (`EXTRACTED`
`mentions`, the row this pass re-types) nor a prior classifier verdict
(`EXTRACTED_CLASSIFIER*`, this pass's own earlier output) is planned as
`protected_pair` and written zero times. `STATED` arrives through `fact_relate`
and `INFERRED` through `conversation_relations` (`co_accessed`) and
`link_stranded_entities`; none of them may be expelled as collateral to promote
a derived row. Measured on a copy of the live store on 2026-09-24: 29,620 pairs
hold an un-judged eligible `mentions` edge, and the store holds no active edge
outside `EXTRACTED`/`EXTRACTED_CLASSIFIER_V4` at all — so the lane plans 0 today
and the guarded state is one `fact_relate` away, not live. A pair carrying a
live classifier verdict of the SAME type is planned `already_typed` rather than
`protected_pair`, even if a human edge is somehow also present: that lane
retires only the redundant `mentions` row, so it expels nothing either.

One apply pass is NOT the fixed point (#1257). A record's eligibility is read
against the index built before the pass: when a pair holds an active eligible
`mentions` edge in BOTH directions and the input carries one orientation's
`related_to` record, the pass consumes the edge in the record's own direction,
and only the next pass — whose index no longer holds that direction — reaches
the reversed-fold branch and retypes the other one. Measured 2026-09-20 on a
copy of the live store: `--apply` wrote 145, the dry-run after it planned 11,
the next `--apply` wrote 11, the next dry-run 0. So `--apply` re-plans and
re-applies until a pass plans zero, bounded at MAX_PASSES; the cap is reported
with the residual plan rather than raised, because the next nightly run picks
it up. What a single retype expires is unchanged — #1246 owns that seam.

Each pass is ONE transaction: a crash halfway through leaves that pass's
graph untouched rather than half-upgraded. A backup is taken before the
first write.

Usage:
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py            # dry-run
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py --apply
  .venvs/lloyd/bin/python scripts/memory/apply-classifications-v4.py --apply --min-confidence 0.75
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import PIPELINE_DIR, VAULT_KG_DB
from app.kg_store import KGStore, canonical_edge_type

CLASSIFIED_DIR = PIPELINE_DIR / "memory-graph"
DEFAULT_GLOB = "classified-v4*.jsonl"
DEFAULT_MIN_CONF = 0.6

# Live edges with these `provenance` values are eligible to be re-typed.
# Anything else (`STATED`, `INFERRED`, prior `EXTRACTED_CLASSIFIER*` outputs)
# represents human or prior-classifier intent we shouldn't override here.
ELIGIBLE_PROVENANCES = frozenset({"EXTRACTED"})

# The classifier family: this pass's own output, whatever prompt version wrote
# it, and the same rule `_build_live_typed_index` already applies. A co-resident
# edge carrying one of these is meant to be superseded by the next verdict; one
# carrying anything else is not (#1453). Measured on a copy of the live store on
# 2026-09-24, every active classifier row is spelled `EXTRACTED_CLASSIFIER_V4`;
# the prefix is kept because the older spelling is what an earlier prompt wrote.
CLASSIFIER_PROVENANCE_PREFIX = "EXTRACTED_CLASSIFIER"

# Apply passes per invocation. Two reach the fixed point on every store
# measured so far (the reversed-fold case above needs exactly one more pass
# than the record count suggests); four is the bound that keeps a store some
# future input could make oscillate from looping forever. Hitting it is
# reported, not raised.
MAX_PASSES = 4


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


def _build_live_typed_index(edges: list[dict]) -> dict[tuple[str, str], dict]:
    """(source, target) → the active edge a prior classifier apply typed it as.

    The extractor used to mint a fresh `mentions` row on a pair whose typed
    verdict was already live (#1246), and that row put the pair back through
    `retype`, which expired the live typed edge and inserted the same type
    again — 3-29% of each apply's output. A record whose `new_type` matches
    this edge is not an upgrade."""
    idx: dict[tuple[str, str], dict] = {}
    for e in edges:
        if e.get("type") == "mentions":
            continue
        if not (e.get("provenance") or "").startswith("EXTRACTED_CLASSIFIER"):
            continue
        idx[(e.get("source", ""), e.get("target", ""))] = e
    return idx


def _build_protected_pairs(edges: list[dict]) -> set[tuple[str, str]]:
    """(source, target) pairs carrying an active edge this pass may not expel.

    `retype` expires EVERY other active edge on the pair whatever its
    provenance, so the "never re-type `STATED`/`INFERRED`" guard rail has to be
    applied here, before a retype is issued, and not inside `retype` — #925
    keeps its one-active-relation-per-pair invariant, and
    `tests/test_kg_store.py::test_retype_keeps_history_and_collapses_the_pair`
    asserts that a co-resident `STATED` row *is* expired when a retype happens
    at all (#1453). A `STATED` row came from a person calling `fact_relate`; an
    `INFERRED` one from `conversation_relations` (`co_accessed`) or
    `link_stranded_entities`.

    Rows this pass is allowed to consume are not blockers: the eligible
    `mentions` edge it is re-typing (the extractor's raw material, retired by
    the retype on purpose), and a prior `EXTRACTED_CLASSIFIER*` verdict, which
    is this pass's own earlier output and reaches its own lanes — `already_typed`
    when the type matches, an upgrade that supersedes it when it does not."""
    protected: set[tuple[str, str]] = set()
    for e in edges:
        prov = e.get("provenance") or ""
        if prov.startswith(CLASSIFIER_PROVENANCE_PREFIX):
            continue
        if e.get("type") == "mentions" and prov in ELIGIBLE_PROVENANCES:
            continue
        protected.add((e.get("source", ""), e.get("target", "")))
    return protected


def _plan(st: KGStore, records: list[dict], min_confidence: float, now: str,
          ) -> tuple[list[dict], dict, Counter, Counter, list[tuple[int, str]]]:
    """One pass's plan against the store AS IT IS NOW.

    Re-read on every pass, which is the whole fix: the index this builds is
    what decides eligibility, and the previous pass's retypes moved it.
    """
    active = st.edges.active()
    active_mentions = _build_active_mentions_index(active)
    live_typed = _build_live_typed_index(active)
    protected_pairs = _build_protected_pairs(active)
    print(f"[info] {st.edges.count(active_only=False)} total edges, {len(active)} active, "
          f"{len(active_mentions)} eligible active mentions edges "
          f"(type=mentions, provenance in {sorted(ELIGIBLE_PROVENANCES)})")
    # The denominator beside any `protected_pair` count below: how many pairs in
    # the whole store hold an edge this pass may not expel, so a `0` lane is
    # readable as "nothing to guard" rather than "the guard is not running".
    print(f"[info] {len(protected_pairs)} active pair(s) carry a non-classifier edge "
          f"(provenance outside {CLASSIFIER_PROVENANCE_PREFIX}*)")

    stats = {
        "total_records": len(records),
        "below_threshold": 0,
        "still_mentions": 0,
        "no_eligible_edge": 0,
        "duplicate_pair": 0,
        "already_typed": 0,
        "protected_pair": 0,
        "upgrades": 0,
    }
    seen_pairs: set[tuple[str, str]] = set()
    transitions: Counter = Counter()
    new_type_counts: Counter = Counter()
    changes: list[dict] = []
    # Redundant `mentions` rows beside a live typed edge of the same type: no
    # retype (the verdict is already live), but not left standing either —
    # `retype`'s one-active-relation-per-pair invariant is deliberate, and a
    # pair with both rows would be counted here again on every run.
    redundant: list[tuple[int, str]] = []

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
        if conf < min_confidence:
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

        if key in protected_pairs:
            # The pair holds an edge this pass did not write. Issuing the
            # retype would expel it as collateral (`reason: … pair re-typed as
            # X`), which is the guard rail this lane exists for, and #925 keeps
            # the one-active-relation-per-pair invariant that produces that
            # sweep — so the pair is left exactly as found and reported instead
            # of silently dropped: it would otherwise read as `upgrades` today
            # and as `no_eligible_edge` after the damage was done (#1453).
            # The extractor already refuses to mint a `mentions` row beside
            # such an edge (#1246); this is the write-path half of that stance.
            stats["protected_pair"] += 1
            continue

        typed = live_typed.get(key)
        if typed is not None and typed.get("type") == canonical_edge_type(new_type):
            stats["already_typed"] += 1
            redundant.append((active_mentions[key], typed["type"]))
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
    return changes, stats, transitions, new_type_counts, redundant


def _print_plan(pass_no: int, stats: dict, transitions: Counter,
                new_type_counts: Counter) -> None:
    print()
    print("=" * 70)
    print(f"v4 reclassification plan (pass {pass_no}):")
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


def _apply(st: KGStore, changes: list[dict], now: str,
           redundant: list[tuple[int, str]] = ()) -> None:
    # One transaction for the whole pass: 3,902 retypes either all land or
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
        for edge_id, typ in redundant:
            st.edges.expire(edge_id, f"v4 already typed as {typ}: redundant mentions row")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--classified-dir", type=Path, default=CLASSIFIED_DIR)
    p.add_argument("--pattern", default=DEFAULT_GLOB,
                   help="Glob inside classified-dir (default: classified-v4*.jsonl)")
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONF)
    p.add_argument("--apply", action="store_true",
                   help="Actually write changes, re-planning until a pass plans "
                        f"zero upgrades (at most {MAX_PASSES} passes)")
    p.add_argument("--dry-run", action="store_true", help="Preview only (default)")
    p.add_argument("--db", type=Path, default=VAULT_KG_DB)
    args = p.parse_args()

    if not args.apply and not args.dry_run:
        print("[info] no --apply specified → dry-run (no writes)")
        args.dry_run = True

    records = _load_v4_records(args.classified_dir, args.pattern)

    st = KGStore(args.db)
    now = datetime.now(timezone.utc).isoformat()
    per_pass: list[int] = []
    backup: Path | None = None

    for pass_no in range(1, MAX_PASSES + 1):
        changes, stats, transitions, new_type_counts, redundant = _plan(
            st, records, args.min_confidence, now)
        _print_plan(pass_no, stats, transitions, new_type_counts)
        per_pass.append(len(changes))

        if args.dry_run:
            print()
            print("[dry-run] no changes written. Re-run with --apply to commit.")
            if changes:
                # A dry run can only show the first pass: what the second pass
                # would plan depends on writes this run is not making.
                print(f"[dry-run] pass 1 plans {len(changes)} upgrades; --apply "
                      f"repeats until a pass plans zero")
            st.close()
            return 0

        if not changes and not redundant:
            break

        if backup is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            backup_dir = args.db.parent / "store-backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = st.backup(backup_dir / f"kg-v4-{ts}.sqlite")
            print(f"\n[info] backup written → {backup}")

        _apply(st, changes, now, redundant)
        print(f"[info] pass {pass_no}: applied {len(changes)} retypes and retired "
              f"{len(redundant)} redundant mentions rows; store now {st.stats()}")
    else:
        # The cap pass wrote, so the residual is whatever a fresh plan says —
        # counted, never applied: the next nightly run takes it.
        residual = len(_plan(st, records, args.min_confidence, now)[0])
        print(f"\n[warn] pass cap reached: passes={MAX_PASSES} upgrades_per_pass={per_pass} "
              f"applied={sum(per_pass)}; {residual} upgrades still planned after the "
              f"cap — re-run --apply to continue")
        st.close()
        return 0

    applied = sum(per_pass)
    if applied == 0:
        print("\n[info] no changes to apply")
    print(f"\n[info] converged: passes={len(per_pass)} upgrades_per_pass={per_pass} "
          f"applied={applied}; last pass planned 0 upgrades")
    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
