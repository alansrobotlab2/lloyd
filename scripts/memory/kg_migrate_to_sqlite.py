#!/usr/bin/env python3
"""Move the knowledge graph from the two JSON blobs into app.kg_store.

Idempotent. Imports `_relationships.json` (every edge, expired ones too) and
`entity-aliases.json` (self-identities become entity rows; the rest are
classified by the sweep's `classify_pair` so every alias says how it differs
from its canonical), registers every entity directory, and rebuilds the fact
index from the markdown files.

The JSON files are left where they are. `--retire-json` renames them to
`*.migrated-<ts>.json` — only after an `export_json` round trip reproduces
the same edge count, active count and alias count, so a reader that was not
converted fails loudly instead of reading a stale file.

Usage:
  .venvs/lloyd/bin/python scripts/memory/kg_migrate_to_sqlite.py               # import + reindex
  .venvs/lloyd/bin/python scripts/memory/kg_migrate_to_sqlite.py --retire-json  # …then rename the JSON
  .venvs/lloyd/bin/python scripts/memory/kg_migrate_to_sqlite.py --db /tmp/x.sqlite --facts-dir /tmp/facts
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))

from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB  # noqa: E402
from app.kg_store import KGStore  # noqa: E402
from _invocation import invocation_ledger  # noqa: E402


def _sweep():
    spec = importlib.util.spec_from_file_location("entity_resolution_sweep", HERE / "entity-resolution-sweep.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("entity_resolution_sweep", mod)
    spec.loader.exec_module(mod)
    return mod


_TIER_TO_KIND = {"CASE": "case", "PUNCT": "punct", "SUFFIX_SAFE": "suffix",
                 "SUFFIX_AMBIGUOUS": "suffix", "IDENTICAL": "self"}


def classify_alias_factory():
    """The sweep's tier is the better classifier when it has an opinion —
    it knows about safe-suffix forms — so use it, and fall back to the
    store's own shape rule for OTHER."""
    sweep = _sweep()
    from app.kg_store import alias_kind

    def classify(surface: str, canonical: str) -> str:
        tier, _ = sweep.classify_pair(surface, canonical)
        return _TIER_TO_KIND.get(tier) or alias_kind(surface, canonical)
    return classify


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, default=VAULT_KG_DB)
    ap.add_argument("--facts-dir", type=Path, default=VAULT_FACTS_ROOT)
    ap.add_argument("--relationships", type=Path, default=None, help="default <facts-dir>/_relationships.json")
    ap.add_argument("--aliases", type=Path, default=None, help="default <facts-dir>/entity-aliases.json")
    ap.add_argument("--skip-reindex", action="store_true")
    ap.add_argument("--retire-json", action="store_true",
                    help="after a verified round trip, rename the JSON files *.migrated-<ts>.json")
    ap.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    args = ap.parse_args()

    facts = args.facts_dir
    rel = args.relationships or (facts / "_relationships.json")
    al = args.aliases or (facts / "entity-aliases.json")
    t0 = time.perf_counter()
    report: dict = {"db": str(args.db), "facts_dir": str(facts), "relationships": str(rel),
                    "aliases": str(al), "ledger": invocation_ledger()}

    print(f"store: {args.db}")
    s = KGStore(args.db)
    before = s.stats()
    print(f"  before: {before}")

    # 1. edges + aliases
    stats = s.import_json(rel if rel.exists() else None, al if al.exists() else None,
                          classify_alias=classify_alias_factory())
    print(f"  import: {stats}")
    report["import"] = stats

    # 2. every entity dir is an entity
    registered = 0
    if facts.exists():
        with s.transaction():
            for d in facts.iterdir():
                if d.is_dir() and not d.name.startswith((".", "_")):
                    if s.entities.register(d.name) is not None:
                        registered += 1
    print(f"  entity dirs registered: {registered}")
    report["entities_registered_from_dirs"] = registered

    # 3. fact index
    if not args.skip_reindex:
        t = time.perf_counter()
        ri = s.facts_idx.reindex(root=facts)
        print(f"  reindex: {ri} in {time.perf_counter() - t:.1f}s")
        report["reindex"] = ri

    after = s.stats()
    print(f"  after: {after}")
    report["before"], report["after"] = before, after

    # 4. round-trip check against the JSON we imported from
    verified = True
    if rel.exists():
        legacy = json.loads(rel.read_text(encoding="utf-8"))["edges"]
        legacy_active = sum(1 for e in legacy if not e.get("expired_at"))
        checks = {
            "legacy_edges": len(legacy), "store_edges": after["edges_total"],
            "legacy_active": legacy_active, "store_active": after["edges_active"],
        }
        if al.exists():
            legacy_al = json.loads(al.read_text(encoding="utf-8"))
            checks["legacy_alias_entries"] = len(legacy_al)
            checks["store_aliases_plus_entities"] = after["aliases"] + after["entities"]
            checks["legacy_non_self_aliases"] = sum(1 for k, v in legacy_al.items() if k != v)
            checks["store_aliases"] = after["aliases"]
        exp_dir = args.db.parent / f"kg-export-check-{dt.datetime.now():%Y%m%dT%H%M%SZ}"
        out = s.export_json(exp_dir)
        exported = json.loads(out["relationships"].read_text())["edges"]
        checks["exported_edges"] = len(exported)
        checks["exported_active"] = sum(1 for e in exported if not e.get("expired_at"))
        ok_edges = checks["store_edges"] >= checks["legacy_edges"] == checks["exported_edges"] or \
            checks["store_edges"] == checks["exported_edges"] >= checks["legacy_edges"]
        ok_active = checks["store_active"] == checks["exported_active"]
        # legacy duplicate-active keys land expired, so active may drop by that many
        dup_expired = sum(1 for e in s.edges.all() if e.get("expired_reason") == "migration: duplicate active edge")
        checks["legacy_duplicate_active_keys"] = dup_expired
        ok_active_vs_legacy = checks["store_active"] + dup_expired >= checks["legacy_active"]
        ok_alias = "store_aliases" not in checks or checks["store_aliases"] >= checks["legacy_non_self_aliases"]
        verified = ok_edges and ok_active and ok_active_vs_legacy and ok_alias
        report["round_trip"] = {**checks, "verified": verified}
        print(f"  round trip: {checks} verified={verified}")
        for p in exp_dir.iterdir():
            p.unlink()
        exp_dir.rmdir()

    # 5. retire the JSON
    if args.retire_json:
        if not verified:
            print("  NOT retiring JSON: round trip did not verify", file=sys.stderr)
            return 2
        ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
        retired = []
        for p in (rel, al):
            if p.exists():
                dest = p.with_name(f"{p.stem}.migrated-{ts}.json")
                p.rename(dest)
                retired.append(str(dest))
        report["retired"] = retired
        print(f"  retired: {retired}")

    s.meta_set("migrated_at", dt.datetime.now(dt.timezone.utc).isoformat())
    report["elapsed_s"] = round(time.perf_counter() - t0, 1)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"  report: {args.report}")
    s.close()
    print(f"done in {report['elapsed_s']}s")
    return 0 if verified else 2


if __name__ == "__main__":
    sys.exit(main())
