#!/usr/bin/env python3
"""Disposition audit over a reverted entity-merge artifact (backlog #475).

`entity-merges-reverted-20260903T174108Z.json` records the 151 variant→canonical
pairs an unattributed `--apply` merged on 2026-09-03 and a revert then undid.
#475 says every one of those pairs must be *dispositioned* before the item can
close: either a later apply put a real apply-origin alias row in the store for
that pair, or the run that looked at it recorded that it declined. Until now no
command could answer that question, which is how alias coverage decayed from
16.4 % to 15.26 % while nobody was touching it.

Read-only. Nothing here merges anything.

What a pair can be:
  applied      an alias row maps the surface to that canonical, written by an
               apply (origin outside the inherited set) and pointing at the
               report that authorized it
  declined     a plan the sweep wrote lists that exact pair with a non-SAFE
               status — the run evaluated it and said no
  unaccounted  neither. Nobody has decided about this pair since the revert.

Exit code: 0 when nothing is unaccounted, 1 when some pairs remain, 2 on a bad
argument or an unopenable store. A store that will not open is never reported as
an empty one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.kg_store import KGStore, StoreUnavailable  # noqa: E402
from app.paths import VAULT_KG_DB  # noqa: E402

OUT_DIR = Path.home() / "lloyd" / "_pipeline" / "memory-graph"

# Origins that arrive from somewhere other than a gated apply: the 2026-09-03
# SQLite migration, the schema-declared naming layer, test fixtures. `revert` is
# the revert tool writing an old mapping back, which is the opposite of applied.
INHERITED_ORIGINS = {"migration", "schema", "test", "revert"}
DECLINED_STATUSES = {"AMBIGUOUS", "SKIPPED"}


def reverted_pairs(artifact: dict) -> list[dict]:
    """The variant→canonical pairs the revert undid, from either shape it has."""
    plan = artifact.get("plan")
    if not isinstance(plan, list):
        plan = (artifact.get("result") or {}).get("file_ops") or []
    out = []
    for e in plan:
        variant, canonical = e.get("variant"), e.get("canonical")
        if variant and canonical:
            out.append({"variant": variant, "canonical": canonical})
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def declined_pairs(store_dir: Path, apply_reports: list[Path],
                  extra_plans: list[Path]) -> dict[tuple[str, str], dict]:
    """Pairs a sweep plan evaluated and refused, keyed by (variant, canonical).

    Only the plans a run actually wrote are consulted — the ones each apply
    report names as its `plan_file`, plus any passed explicitly. A plan from a
    run that never applied is not evidence about a run that did.
    """
    plans: list[Path] = list(extra_plans)
    for rep in apply_reports:
        try:
            pf = json.loads(rep.read_text(encoding="utf-8")).get("plan_file")
        except (OSError, json.JSONDecodeError):
            continue
        if pf:
            plans.append(Path(pf))
    out: dict[tuple[str, str], dict] = {}
    for plan in plans:
        if not plan or not Path(plan).is_file():
            continue
        for cluster in _read_jsonl(Path(plan)):
            if cluster.get("status") not in DECLINED_STATUSES:
                continue
            names = {v[0] for v in cluster.get("variants", []) if isinstance(v, (list, tuple))}
            names |= {v["variant"] for v in cluster.get("merges", [])
                      if isinstance(v, dict) and v.get("variant")}
            canonical = cluster.get("canonical")
            for name in names:
                if canonical:
                    out.setdefault((name, canonical),
                                   {"status": cluster["status"], "plan": str(plan),
                                    "decision": cluster.get("decision", "")})
                for other in names - {name}:
                    out.setdefault((name, other),
                                   {"status": cluster["status"], "plan": str(plan),
                                    "decision": cluster.get("decision", "")})
    return out


def audit(pairs: list[dict], alias_rows: dict[str, list[dict]],
          declined: dict[tuple[str, str], dict]) -> dict:
    """Classify every pair. `alias_rows` is surface → rows, from the live store.

    A surface may carry several alias rows (one per canonical it has been mapped
    to over time), so a pair counts as applied only when *a row for that exact
    canonical* was written by an apply — an inherited row pointing elsewhere is
    not a disposition, it is the fragmentation this item exists to remove.
    """
    entries, counts = [], {"applied": 0, "declined": 0, "unaccounted": 0}
    for p in pairs:
        variant, canonical = p["variant"], p["canonical"]
        rows = alias_rows.get(variant, [])
        applied = next((r for r in rows if r.get("canonical") == canonical
                        and r.get("origin") not in INHERITED_ORIGINS
                        and r.get("origin") is not None), None)
        if applied is not None:
            status = "applied"
            report = applied.get("report_path")
            detail = {"origin": applied.get("origin"), "report_path": report}
            if not report:
                detail["provenance_missing"] = True
            elif not Path(report).is_file():
                detail["report_missing"] = True
        elif (variant, canonical) in declined:
            d = declined[(variant, canonical)]
            status, detail = "declined", {"status": d["status"], "plan": d["plan"],
                                          "decision": d["decision"][:200]}
        else:
            status = "unaccounted"
            detail = {}
            inherited = next((r for r in rows if r.get("origin") in INHERITED_ORIGINS), None)
            if inherited:
                detail["inherited_alias"] = {"canonical": inherited.get("canonical"),
                                             "origin": inherited.get("origin")}
            for (v, c), d in declined.items():
                if v == variant:
                    detail["variant_declined_under"] = {"canonical": c,
                                                       "status": d["status"]}
                    break
        counts[status] += 1
        entries.append({"variant": variant, "canonical": canonical,
                        "status": status, **({"detail": detail} if detail else {})})
    return {"counts": counts, "unaccounted": counts["unaccounted"], "entries": entries}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifact", default=None,
                    help="reverted-merges JSON (default: newest entity-merges-reverted-*.json in --out-dir)")
    ap.add_argument("--db", default=str(VAULT_KG_DB), help="KG store to read (read-only)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="where apply reports live")
    ap.add_argument("--plan", action="append", default=[],
                    help="extra plan .jsonl to treat as declination evidence (repeatable)")
    ap.add_argument("--out", default=None, help="where to write the audit JSON")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    artifact_path = (Path(args.artifact) if args.artifact else
                     max(out_dir.glob("entity-merges-reverted-*.json"), key=lambda p: p.stat().st_mtime,
                         default=None))
    if not artifact_path or not artifact_path.is_file():
        print(f"no reverted-merges artifact found in {out_dir}", file=sys.stderr)
        return 2
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"cannot read {artifact_path}: {e}", file=sys.stderr)
        return 2

    pairs = reverted_pairs(artifact)
    if not pairs:
        print(f"{artifact_path} lists no variant→canonical pairs", file=sys.stderr)
        return 2

    try:
        st = KGStore(Path(args.db))
    except StoreUnavailable as e:
        print(f"store will not open ({args.db}): {e}", file=sys.stderr)
        return 2
    apply_reports = sorted(out_dir.glob("entity-merges-applied-*.json"))
    try:
        alias_rows: dict[str, list[dict]] = {}
        for r in st.aliases.rows():
            alias_rows.setdefault(r["surface"], []).append(r)
        report: dict = {
            "artifact": str(artifact_path),
            "db": str(args.db),
            "pairs": len(pairs),
            "apply_reports_scanned": [str(p) for p in apply_reports],
        }
        report.update(audit(pairs, alias_rows,
                            declined_pairs(out_dir, apply_reports,
                                           [Path(p) for p in args.plan])))
    finally:
        st.close()

    report["checked_at"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = Path(args.out) if args.out else out_dir / (
        f"entity-merge-disposition-{dt.date.today().isoformat()}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    c = report["counts"]
    print(f"artifact:      {report['artifact']}")
    print(f"pairs:         {report['pairs']}")
    print(f"  applied:     {c['applied']}")
    print(f"  declined:    {c['declined']}")
    print(f"  UNACCOUNTED: {c['unaccounted']}")
    print(f"report:        {out}")
    if c["unaccounted"]:
        for e in report["entries"]:
            if e["status"] == "unaccounted":
                why = e.get("detail", {})
                note = ""
                if "inherited_alias" in why:
                    ia = why["inherited_alias"]
                    note = f"  (alias exists: → {ia['canonical']!r} origin={ia['origin']})"
                if "variant_declined_under" in why:
                    vd = why["variant_declined_under"]
                    note += f"  (declined under canonical {vd['canonical']!r})"
                print(f"  ? {e['variant']!r} → {e['canonical']!r}{note}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
