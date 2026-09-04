"""Knowledge-graph hygiene metrics. READ-ONLY.

Three things the 2026-09-03 audit had to compute by hand, now measured on
demand and by kg_health.py:

  contamination   entity directories holding facts whose own `entity:` tag names
                  a DIFFERENT entity — the residue of a wrong merge. Every one of
                  the 63 found on 2026-09-03 came from the sweep's suffix tier
                  fusing distinct things (`Intel Pipeline System` into `Intel`).
  near_duplicates directories that collapse to the same name after
                  normalisation, by tier — the sweep's input.
  regrowth        near-duplicate directories born in the last N days next to an
                  older one — the rate at which extraction re-creates duplicates,
                  i.e. the number extraction-time linking is meant to drive down.

Usage:
  python scripts/memory/kg_hygiene.py            # human summary
  python scripts/memory/kg_hygiene.py --json     # raw JSON
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from app.paths import VAULT_FACTS_ROOT  # noqa: E402

_SWEEP_PATH = Path(__file__).resolve().parent / "entity-resolution-sweep.py"
_sweep_mod = None


def sweep():
    """The sweep script owns the name normalisers; load it once by path."""
    global _sweep_mod
    if _sweep_mod is None:
        spec = importlib.util.spec_from_file_location("entity_resolution_sweep", _SWEEP_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["entity_resolution_sweep"] = mod
        spec.loader.exec_module(mod)
        _sweep_mod = mod
    return _sweep_mod


def iter_entity_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))


def parse_frontmatter(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        fm = yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}
    return fm if isinstance(fm, dict) else {}


def _same_entity(a: str, b: str) -> bool:
    s = sweep()
    return s.normalize_punct(a) == s.normalize_punct(b)


# ── contamination ─────────────────────────────────────────────────────────────

def foreign_facts_in_dir(d: Path) -> dict[str, dict[str, Any]]:
    """{foreign_entity: {"facts": n, "files": [...]}} for one entity directory."""
    out: dict[str, dict[str, Any]] = {}
    for f in d.glob("*.md"):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        names: Counter[str] = Counter()
        top = str(fm.get("entity") or "").strip()
        facts = fm.get("facts") or []
        if isinstance(facts, list) and facts:
            for x in facts:
                if isinstance(x, dict) and x.get("entity"):
                    names[str(x["entity"]).strip()] += 1
        elif top:
            names[top] += 1          # overview / factless file: the file-level tag
        for name, n in names.items():
            if not name or _same_entity(name, d.name):
                continue
            slot = out.setdefault(name, {"facts": 0, "files": []})
            slot["facts"] += n
            if f.name not in slot["files"]:
                slot["files"].append(f.name)
    return out


def contamination(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    s = sweep()
    items = []
    by_tier: Counter[str] = Counter()
    total_facts = 0
    for d in iter_entity_dirs(root):
        foreign = foreign_facts_in_dir(d)
        if not foreign:
            continue
        for name, slot in foreign.items():
            by_tier[s.classify_pair(name, d.name)[0]] += 1
            total_facts += slot["facts"]
        items.append({"dir": d.name, "foreign": foreign})
    return {"dirs": len(items), "foreign_facts": total_facts,
            "by_tier": dict(by_tier), "items": items}


# ── near-duplicates and regrowth ──────────────────────────────────────────────

def _clusters(root: Path) -> dict[str, list[Path]]:
    s = sweep()
    by: dict[str, list[Path]] = defaultdict(list)
    for d in iter_entity_dirs(root):
        by[s.normalize_full(d.name)].append(d)
    return {k: v for k, v in by.items() if len(v) > 1}


def near_duplicates(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    s = sweep()
    cl = _clusters(root)
    tiers = Counter(s.cluster_tier([d.name for d in v]) for v in cl.values())
    return {"clusters": len(cl), "dirs": sum(len(v) for v in cl.values()),
            "by_tier": dict(tiers),
            "samples": [[d.name for d in v] for v in list(cl.values())[:8]]}


_TS_KEYS = ("created_at", "created", "first_seen", "timestamp")


def _parse_ts(v) -> float | None:
    if isinstance(v, dt.datetime):
        return (v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)).timestamp()
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day, tzinfo=dt.timezone.utc).timestamp()
    try:
        t = dt.datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).timestamp()
    except (TypeError, ValueError):
        return None


def _born(d: Path) -> float:
    """When this entity first existed: the earliest fact `created_at` in the
    directory, falling back to file mtime. mtime alone is unreliable — a bulk
    revert, merge or retag rewrites every file and makes a year-old entity look
    born today (6,625 "new" dirs after the 2026-09-03 repairs)."""
    best: float | None = None
    mtimes = []
    for f in d.glob("*.md"):
        try:
            mtimes.append(f.stat().st_mtime)
        except OSError:
            continue
        fm = parse_frontmatter(f)
        cands = [fm] + [x for x in (fm.get("facts") or []) if isinstance(x, dict)]
        for x in cands:
            for k in _TS_KEYS:
                if x.get(k) is not None:
                    t = _parse_ts(x[k])
                    if t is not None and (best is None or t < best):
                        best = t
    if best is not None:
        return best
    return min(mtimes) if mtimes else d.stat().st_mtime


def regrowth(root: Path = VAULT_FACTS_ROOT, days: int = 7,
             now: float | None = None) -> dict[str, Any]:
    s = sweep()
    now = now or dt.datetime.now().timestamp()
    born = {d.name: _born(d) for d in iter_entity_dirs(root)}
    by: dict[str, list[str]] = defaultdict(list)
    for n in born:
        by[s.normalize_full(n)].append(n)
    new = [n for n, t in born.items() if now - t < days * 86400]
    dups, tiers = [], Counter()
    for n in new:
        older = [o for o in by[s.normalize_full(n)] if o != n and born[o] < born[n] - 3600]
        if older:
            dups.append(n)
            tiers[s.classify_pair(n, older[0])[0]] += 1
    return {"days": days, "new_dirs": len(new), "near_dup_new": len(dups),
            "by_tier": dict(tiers), "samples": dups[:8]}


def snapshot(root: Path = VAULT_FACTS_ROOT, days: int = 7) -> dict[str, Any]:
    c = contamination(root)
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "facts_root": str(root),
        "contamination": {k: v for k, v in c.items() if k != "items"},
        "near_duplicates": near_duplicates(root),
        "regrowth": regrowth(root, days),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--items", action="store_true", help="list contaminated dirs")
    args = ap.parse_args()
    if args.items:
        c = contamination()
        for it in c["items"]:
            for name, slot in it["foreign"].items():
                print(f"{it['dir']!r:40} holds {slot['facts']:>3} fact(s) for {name!r}  ({', '.join(slot['files'])})")
        print(f"\n{c['dirs']} dirs, {c['foreign_facts']} foreign facts, by tier {c['by_tier']}")
        return
    s = snapshot(days=args.days)
    if args.json:
        print(json.dumps(s, indent=2))
        return
    c, n, r = s["contamination"], s["near_duplicates"], s["regrowth"]
    print("Knowledge-graph hygiene")
    print(f"  contamination   {c['dirs']:>6} dirs hold {c['foreign_facts']} facts about another entity  {c['by_tier']}")
    print(f"  near-duplicates {n['clusters']:>6} clusters over {n['dirs']} dirs  {n['by_tier']}")
    print(f"  regrowth {r['days']}d     {r['near_dup_new']:>6} of {r['new_dirs']} new dirs are near-dups of an older one  {r['by_tier']}")


if __name__ == "__main__":
    main()
