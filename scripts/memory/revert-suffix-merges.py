#!/usr/bin/env python3
"""Undo entity merges recorded by an entity-resolution-sweep apply report.

Dry-run by default. Built for the 2026-09-03 12:32Z apply, which merged 151
suffix pairs against an empty graph — `Intel Pipeline System` into `Intel`,
`Triage Agent` into `TRIAGE` — but general: point it at any apply report and a
tier, and it puts the variant's facts back in the variant's own directory.

What it inverts, per (variant → canonical) in the report:
  * a fact file whose facts all belong to the variant is moved back whole and
    its filename prefix restored;
  * a fact file the sweep MERGED is split by each fact's own `entity:` tag —
    the variant's facts go to `<variant>/<variant>-<category>.md`, the rest
    stay, and an emptied canonical file is removed;
  * a variant overview that was renamed into the canonical dir is moved back
    (one the sweep DISCARDED on collision is gone and is reported as such);
  * alias entries routing the variant to the canonical are removed and the
    variant re-registered as itself, so the extractor stops filing new facts
    under the wrong name.

`--fix-edges` handles the follow-on damage: edges seeded from the canonical
directory while it still held the variant's prose. For every touched
canonical, any active seeded edge whose target is no longer named anywhere in
the canonical's OWN fact text is expired. Re-run the seeder afterwards to
attach those relationships to the restored variant.

Usage:
  revert-suffix-merges.py --applied <report.json>                 # dry-run
  revert-suffix-merges.py --applied <report.json> --apply
  revert-suffix-merges.py --applied <report.json> --apply --fix-edges
  revert-suffix-merges.py --applied <report.json> --tiers SUFFIX_SAFE,SUFFIX_AMBIGUOUS
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
from app.paths import VAULT_FACTS_ROOT, VAULT_FACTS_ALIASES  # noqa: E402
from _invocation import invocation_ledger  # noqa: E402
import kg_hygiene  # noqa: E402

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _dump(fm: dict, body: str) -> str:
    return f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False, allow_unicode=True)}---\n\n{body}"


def _body(entity: str, category: str, n: int) -> str:
    return f"\n# {entity} - {category}\n\n**Entity:** {entity}\n**Category:** {category}\n**Fact Count:** {n}\n"


def _read(path: Path) -> tuple[dict, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        fm = {}
    return (fm if isinstance(fm, dict) else {}), parts[2].lstrip("\n")


def _same(a: str, b: str) -> bool:
    s = kg_hygiene.sweep()
    return s.normalize_punct(a) == s.normalize_punct(b)


def _belongs(fact: dict, top: str, variant: str) -> bool:
    """A fact is the variant's if it is tagged with the variant, or was retagged
    to the canonical by a merge that recorded `merged_from: <variant>`."""
    return _same(str(fact.get("entity") or top), variant) or _same(str(fact.get("merged_from") or ""), variant)


def _merge_facts(existing: list, incoming: list) -> list:
    seen = {(f.get("fact") or "").strip().lower() for f in existing if isinstance(f, dict)}
    out = list(existing)
    for f in incoming:
        t = (f.get("fact") or "").strip().lower()
        if t and t not in seen:
            out.append(f); seen.add(t)
    return out


def plan_revert(report: dict, tiers: set[str], root: Path) -> list[dict]:
    """One op-list entry per (variant, canonical) pair in the selected tiers."""
    s = kg_hygiene.sweep()
    ops: list[dict] = []
    for variant, canonical in report.get("variant_to_canonical", {}).items():
        if s.classify_pair(variant, canonical)[0] not in tiers:
            continue
        cdir = root / canonical
        entry: dict[str, Any] = {"variant": variant, "canonical": canonical,
                                 "files": [], "lost_overview": False}
        if not cdir.is_dir():
            entry["note"] = "canonical dir missing"
            ops.append(entry); continue
        for f in sorted(cdir.glob("*.md")):
            fm, _ = _read(f)
            if not fm:
                continue
            ftype = fm.get("type")
            top = str(fm.get("entity") or "").strip()
            facts = [x for x in (fm.get("facts") or []) if isinstance(x, dict)]
            if ftype == "overview" or not facts:
                if top and _same(top, variant):
                    entry["files"].append({"file": f.name, "action": "move_overview"})
                continue
            mine = [x for x in facts if _belongs(x, top, variant)]
            if not mine:
                continue
            if len(mine) == len(facts) and (not top or _same(top, variant) or all(x.get("merged_from") for x in mine)):
                entry["files"].append({"file": f.name, "action": "move_whole", "facts": len(mine)})
            else:
                entry["files"].append({"file": f.name, "action": "split",
                                       "facts": len(mine), "remaining": len(facts) - len(mine)})
        if not any(o["action"] == "move_overview" for o in entry["files"]):
            # the sweep keeps the canonical's overview and discards the variant's
            entry["lost_overview"] = True
        ops.append(entry)
    return ops


def _variant_filename(name: str, canonical: str, variant: str) -> str:
    """`Intel-goal.md` -> `Intel Pipeline System-goal.md`."""
    if name.startswith(canonical + "-"):
        return variant + name[len(canonical):]
    if name.startswith(variant + "-"):
        return name
    return f"{variant}-{name}"


def execute(ops: list[dict], root: Path, aliases_path: Path, apply: bool) -> dict:
    done: list[dict] = []
    touched_canonicals: set[str] = set()
    for entry in ops:
        variant, canonical = entry["variant"], entry["canonical"]
        cdir, vdir = root / canonical, root / variant
        for op in entry["files"]:
            src = cdir / op["file"]
            if not src.exists():
                continue
            fm, body = _read(src)
            category = str(fm.get("category") or "general")
            dest = vdir / _variant_filename(src.name, canonical, variant)
            rec = {"variant": variant, "canonical": canonical, "action": op["action"],
                   "from": str(src.relative_to(root)), "to": str(dest.relative_to(root))}
            if not apply:
                done.append(rec); continue
            vdir.mkdir(parents=True, exist_ok=True)
            if op["action"] in ("move_whole", "move_overview"):
                fm["entity"] = variant
                for x in (fm.get("facts") or []):
                    if isinstance(x, dict):
                        x["entity"] = variant
                        x.pop("merged_from", None)
                if dest.exists():
                    dfm, _ = _read(dest)
                    dfm["facts"] = _merge_facts(dfm.get("facts") or [], fm.get("facts") or [])
                    dfm["entity"] = variant
                    dest.write_text(_dump(dfm, _body(variant, category, len(dfm["facts"]))), encoding="utf-8")
                    src.unlink()
                else:
                    new_body = body if fm.get("type") == "overview" else _body(variant, category, len(fm.get("facts") or []))
                    dest.write_text(_dump(fm, new_body), encoding="utf-8")
                    src.unlink()
            else:  # split
                facts = [x for x in (fm.get("facts") or []) if isinstance(x, dict)]
                top = str(fm.get("entity") or "")
                mine = [x for x in facts if _belongs(x, top, variant)]
                rest = [x for x in facts if x not in mine]
                for x in mine:
                    x["entity"] = variant
                    x.pop("merged_from", None)
                if dest.exists():
                    dfm, _ = _read(dest)
                    dfm["facts"] = _merge_facts(dfm.get("facts") or [], mine)
                else:
                    dfm = {"type": "facts", "entity": variant, "category": category, "facts": mine}
                dfm["entity"] = variant
                dfm["last_updated"] = dt.datetime.now().isoformat()
                dest.write_text(_dump(dfm, _body(variant, category, len(dfm["facts"]))), encoding="utf-8")
                if rest:
                    fm["facts"] = rest
                    fm["last_updated"] = dt.datetime.now().isoformat()
                    src.write_text(_dump(fm, _body(canonical, category, len(rest))), encoding="utf-8")
                else:
                    src.unlink()
                    rec["removed_emptied_canonical_file"] = True
            done.append(rec)
            touched_canonicals.add(canonical)

    # aliases: stop routing the variant to the canonical; make the variant itself again
    alias_ops = []
    if aliases_path.exists():
        aliases = json.loads(aliases_path.read_text(encoding="utf-8"))
        for entry in ops:
            v, c = entry["variant"], entry["canonical"]
            for k in list(aliases):
                if aliases[k] == c and _same(k, v):
                    alias_ops.append({"remove": k, "was": c}); del aliases[k]
            if v not in aliases:
                alias_ops.append({"add": v}); aliases[v] = v
        if apply and alias_ops:
            bak = aliases_path.with_suffix(aliases_path.suffix + f".{dt.datetime.now():%Y%m%dT%H%M%SZ}.revert.bak")
            shutil.copy2(aliases_path, bak)
            aliases_path.write_text(json.dumps(aliases, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return {"file_ops": done, "alias_ops": alias_ops, "touched_canonicals": sorted(touched_canonicals)}


def _own_prose(d: Path) -> str:
    out = []
    for f in d.glob("*.md"):
        fm, body = _read(f)
        for x in (fm.get("facts") or []):
            if isinstance(x, dict) and x.get("fact"):
                out.append(str(x["fact"]))
        if fm.get("definition"):
            out.append(str(fm["definition"]))
        out.append(body)
    return "\n".join(out).lower()


def fix_edges(rel_path: Path, canonicals: list[str], root: Path, since: str, apply: bool) -> dict:
    """Expire seeded edges from a canonical that its own remaining text no longer supports."""
    data = json.loads(rel_path.read_text(encoding="utf-8"))
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    expired = []
    prose_cache: dict[str, str] = {}
    for e in data["edges"]:
        if e.get("expired_at") or e.get("source") not in canonicals:
            continue
        if e.get("provenance") not in ("EXTRACTED", "EXTRACTED_CLASSIFIER_V4"):
            continue
        if str(e.get("created_at") or "") < since:
            continue
        src = e["source"]
        prose = prose_cache.setdefault(src, _own_prose(root / src) if (root / src).is_dir() else "")
        tl = str(e.get("target") or "").lower()
        if tl and re.search(r"(?<![a-z0-9])" + re.escape(tl) + r"(?![a-z0-9])", prose):
            continue
        expired.append({"source": src, "target": e.get("target"), "type": e.get("type")})
        if apply:
            e["expired_at"] = now
            e["expired_reason"] = "revert-suffix-merges: target no longer named in source's own facts"
    if apply and expired:
        bak = rel_path.with_suffix(f".{dt.datetime.now():%Y%m%dT%H%M%SZ}.revert.bak.json")
        shutil.copy2(rel_path, bak)
        rel_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"expired": expired, "count": len(expired)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--applied", required=True, type=Path, help="entity-merges-applied-*.json")
    ap.add_argument("--tiers", default="SUFFIX_SAFE", help="comma list of tiers to revert")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--fix-edges", action="store_true")
    ap.add_argument("--facts-dir", type=Path, default=VAULT_FACTS_ROOT)
    ap.add_argument("--aliases", type=Path, default=VAULT_FACTS_ALIASES)
    ap.add_argument("--relationships", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path.home() / "lloyd" / "_pipeline" / "memory-graph")
    args = ap.parse_args()

    report = json.loads(args.applied.read_text(encoding="utf-8"))
    tiers = {t.strip() for t in args.tiers.split(",") if t.strip()}
    root = args.facts_dir
    rel_path = args.relationships or (root / "_relationships.json")

    ops = plan_revert(report, tiers, root)
    n_files = sum(len(o["files"]) for o in ops)
    lost = sum(1 for o in ops if o["lost_overview"])
    print(f"== Revert plan: {len(ops)} merges in tiers {sorted(tiers)} → {n_files} file operations, "
          f"{lost} variant overview(s) unrecoverable ==")
    for o in ops:
        for f in o["files"]:
            print(f"  {o['canonical']!r}/{f['file']}  -{f['action']}->  {o['variant']!r}"
                  + (f"  ({f.get('facts')} facts)" if f.get("facts") else ""))
    if not args.apply:
        print("\n(dry-run — pass --apply to execute)")
        return 0

    result = execute(ops, root, args.aliases, apply=True)
    print(f"\nApplied {len(result['file_ops'])} file ops, {len(result['alias_ops'])} alias ops")
    edges = None
    if args.fix_edges and rel_path.exists():
        since = report.get("ledger", {}).get("timestamp") or "2026-09-03T00:00:00"
        edges = fix_edges(rel_path, result["touched_canonicals"], root, since, apply=True)
        print(f"Expired {edges['count']} seeded edge(s) no longer supported by their source's own facts")

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"entity-merges-reverted-{ts}.json"
    out.write_text(json.dumps({"applied_report": str(args.applied), "tiers": sorted(tiers),
                               "plan": ops, "result": result, "edges": edges,
                               "ledger": invocation_ledger()}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Record: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
