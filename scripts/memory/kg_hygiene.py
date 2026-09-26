"""Knowledge-graph hygiene metrics. READ-ONLY over the facts root.

Three things the 2026-09-03 audit had to compute by hand, now measured on
demand and by kg_health.py:

  contamination   entity directories holding facts whose own `entity:` tag names
                  a DIFFERENT entity — the residue of a wrong merge. Every one of
                  the 63 found on 2026-09-03 came from the sweep's suffix tier
                  fusing distinct things (`Intel Pipeline System` into `Intel`).
  near_duplicates directories that collapse to the same name after
                  normalisation, by tier — the sweep's input.
  regrowth        directories created since a STORED BASELINE of the entity-dir
                  name set, and how many of them are near-dups of a name that
                  already existed at that baseline — the rate at which
                  extraction re-creates duplicates, i.e. the number
                  extraction-time linking is meant to drive down.

The only file this module writes is that baseline (`write_baseline`, under the
data root, never under the facts root). It exists because dating dirs was not
measurable: every rebuilt `created_at` sits behind the whole tree, so a window
could not tell new from old (#1535, in `regrowth` below).

Usage:
  python scripts/memory/kg_hygiene.py                 # human summary
  python scripts/memory/kg_hygiene.py --json          # raw JSON
  python scripts/memory/kg_hygiene.py --write-baseline  # store today's dir set as the reference
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
from app.paths import PIPELINE_DIR, VAULT_FACTS_ROOT  # noqa: E402
from app.kg_store import StoreUnavailable, store  # noqa: E402

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


# ── the baseline regrowth diffs against (#1535) ───────────────────────────────

#: The stored reference for `regrowth()`: next to the kg_health snapshots it
#: feeds, under the data root. Not the repo — `data/**` is a human-only path —
#: and not the facts root, which this module never writes.
BASELINE_PATH = PIPELINE_DIR / "metrics" / "kg-entity-dirs-baseline.json"

BASELINE_SCHEMA = 1


def write_baseline(root: Path = VAULT_FACTS_ROOT,
                   baseline_path: Path | str | None = None) -> dict[str, Any]:
    """Store the current set of entity-directory NAMES as regrowth's reference.

    The name set, deliberately, and not their timestamps. The 2026-09-23 KG
    rebuild re-derived every fact's `created_at`, so nothing in the tree dated
    older than three days and a 7-day window necessarily covered all of it:
    `new_dirs` read 11,959 of 11,959 and 12,027 of 12,027 on the snapshots
    taken after it (#1535). A rebuild can re-date a directory; it cannot make a
    name that was absent from a stored set have been present in it.

    Atomic replace, because the reader is a health run in the same minute: a
    truncated list here would make every directory look new.
    """
    p = Path(baseline_path) if baseline_path else BASELINE_PATH
    rec = {"schema": BASELINE_SCHEMA,
           "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
           "facts_root": str(Path(root).resolve()),
           "dirs": sorted(d.name for d in iter_entity_dirs(root))}
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(p)
    return {"path": str(p), "dirs": len(rec["dirs"]), "captured_at": rec["captured_at"]}


def _read_baseline(root: Path, path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """(baseline, why-not). Never guesses: an unusable reference is reported as
    one by `regrowth`, which is the whole point of having a reference.

    A baseline taken over a DIFFERENT tree is rejected rather than diffed
    against — the report runs over whatever `--facts-dir` names, and subtracting
    some other tree's directory set would print a confident wrong number, which
    is the defect this section was rewritten to stop printing."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, f"no baseline file at {path} (run `kg_health.py -o` or `kg_hygiene.py --write-baseline`)"
    except (OSError, ValueError) as e:
        return None, f"baseline at {path} is unreadable: {type(e).__name__}: {e}"
    dirs = raw.get("dirs") if isinstance(raw, dict) else None
    if not isinstance(dirs, list) or not dirs:
        return None, f"baseline at {path} names no directories"
    taken_over = str(raw.get("facts_root") or "")
    if taken_over and Path(taken_over).resolve() != Path(root).resolve():
        return None, f"baseline at {path} was taken over {taken_over}, not {Path(root).resolve()}"
    return ({"dirs": [str(x) for x in dirs],
             "captured_at": raw.get("captured_at"),
             "schema": raw.get("schema")}, None)


def regrowth(root: Path = VAULT_FACTS_ROOT, days: int = 7,
             baseline_path: Path | str | None = None) -> dict[str, Any]:
    """Entity directories created since the stored baseline of that name set.

    `new_dirs` is a set difference against the baseline `write_baseline`
    stores, never a count of directories whose `created_at`/mtime falls inside a
    window. A window over mutable timestamps is unfalsifiable: after the
    2026-09-23 rebuild every stamp sat behind the whole tree, so all 12,027
    directories were "born in the last 7 days" and `new_dirs` equalled
    `entities.count` on every snapshot — no amount of contamination regrowth
    could have made it read differently, and a reader took it as "the store grew
    by 12,027 this week" (#1535). With no usable baseline this section reports
    `new_dirs: None` and `no_baseline_reason` rather than the size of the store.

    The number carries its own denominator: `dirs_at_baseline` over
    `baseline_at`, both from the same `iter_entity_dirs` walk that produced the
    live set, so it can never disagree with `entities.count` the way two
    separate walks did (11,812 against 11,807 on 2026-09-25). `days` stays in
    the section for continuity with every snapshot already on disk; it bounds
    nothing now, which is why no printed line quotes it.

    `near_dup_new` counts the baseline-new directories whose normalised name
    equals one that was ALREADY there at the baseline — extraction coining a
    twin next to an existing entity. Two twins that both postdate the baseline
    are not regrowth: neither one is the older. `samples` names those twins,
    which is what the report renders as the regrown pairs.
    """
    s = sweep()
    path = Path(baseline_path) if baseline_path else BASELINE_PATH
    base, why = _read_baseline(root, path)
    live = iter_entity_dirs(root)
    if base is None:
        return {"days": days, "new_dirs": None, "near_dup_new": None,
                "by_tier": {}, "samples": [], "skipped_vanished": 0,
                "dirs_at_baseline": None, "baseline_at": None,
                "baseline_path": str(path), "no_baseline_reason": why}
    at_baseline = set(base["dirs"])
    older_by: dict[str, list[str]] = defaultdict(list)
    for name in at_baseline:
        older_by[s.normalize_full(name)].append(name)

    new: list[str] = []
    vanished = 0
    for d in live:
        if d.name in at_baseline:
            continue
        # Re-check the dirs this is about to call new: the entity sweep renames
        # directories while the pass runs, and a monitor that dies on its
        # subject being edited reports nothing at all (#1404).
        try:
            still_there = d.is_dir()
        except OSError:
            still_there = False
        if not still_there:
            vanished += 1
            continue
        new.append(d.name)

    dups: list[str] = []
    tiers: Counter[str] = Counter()
    for n in new:
        older = sorted(o for o in older_by.get(s.normalize_full(n), []) if o != n)
        if older:
            dups.append(n)
            tiers[s.classify_pair(n, older[0])[0]] += 1

    return {"days": days, "new_dirs": len(new), "near_dup_new": len(dups),
            "by_tier": dict(tiers), "samples": dups[:8],
            # A dir renamed mid-pass is skipped, not fatal; counted so a mass
            # skip reads as one and not as a quietly smaller denominator.
            "skipped_vanished": vanished,
            "dirs_at_baseline": len(at_baseline), "baseline_at": base["captured_at"],
            "baseline_path": str(path), "no_baseline_reason": None}


def describe(r: dict[str, Any]) -> str:
    """The counts and the reference they are a delta against, in one phrase.

    Every human surface that prints regrowth — kg_health's summary, the
    knowledge report's Hygiene row, this module's own summary — goes through
    here, because what #1535 actually leaked was a bare `of 12027 new dirs` on
    two of them with nothing beside it to say what that 12,027 was a count of.
    A snapshot written before the baseline existed renders as well, and says
    its number has no recorded reference instead of presenting it as a delta.
    """
    new = r.get("new_dirs")
    if new is None:
        return f"not measured: {r.get('no_baseline_reason') or 'no baseline was named'}"
    near = r.get("near_dup_new")
    at, dirs_at = r.get("baseline_at"), r.get("dirs_at_baseline")
    if at and isinstance(dirs_at, int):
        ref = f"baseline {at} over {dirs_at:,} dirs"
    else:
        ref = "no baseline recorded — this snapshot predates the entity-dir baseline (#1535)"
    return f"{near if near is not None else '?'} of {new:,} new dirs ({ref})"


def snapshot(root: Path = VAULT_FACTS_ROOT, days: int = 7,
             baseline_path: Path | str | None = None) -> dict[str, Any]:
    c = contamination(root)
    out = {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "facts_root": str(root),
        "contamination": {k: v for k, v in c.items() if k != "items"},
        "near_duplicates": near_duplicates(root),
        "regrowth": regrowth(root, days, baseline_path=baseline_path),
    }
    out["provenance"] = provenance_coverage()
    return out


def provenance_coverage(root: Path = VAULT_FACTS_ROOT) -> dict[str, Any]:
    """Share of indexed facts that say where they came from and when.

    99.7% of the 205k facts in the pre-rebuild tree had neither `created_at`
    nor `source_doc`, so nothing could be dated, attributed or selectively
    reverted. The rebuild gate requires 100%.
    """
    try:
        st = store()
        total = st.facts_idx.count()
        if not total:
            return {"facts": 0, "created_at_pct": 0.0, "source_doc_pct": 0.0, "both_pct": 0.0}
        rows = st._query(
            "SELECT SUM(created_at IS NOT NULL) AS c, SUM(source_doc IS NOT NULL) AS s, "
            "SUM(created_at IS NOT NULL AND source_doc IS NOT NULL) AS b FROM facts_idx")[0]
        pct = lambda n: round(100.0 * (n or 0) / total, 2)  # noqa: E731
        return {"facts": total, "created_at_pct": pct(rows["c"]),
                "source_doc_pct": pct(rows["s"]), "both_pct": pct(rows["b"])}
    except StoreUnavailable as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--items", action="store_true", help="list contaminated dirs")
    ap.add_argument("--baseline", type=Path, default=None,
                    help=f"entity-dir baseline to diff against (default: {BASELINE_PATH})")
    ap.add_argument("--write-baseline", action="store_true",
                    help="store today's entity-dir set as the reference regrowth diffs against, and stop")
    args = ap.parse_args()
    if args.items:
        c = contamination()
        for it in c["items"]:
            for name, slot in it["foreign"].items():
                print(f"{it['dir']!r:40} holds {slot['facts']:>3} fact(s) for {name!r}  ({', '.join(slot['files'])})")
        print(f"\n{c['dirs']} dirs, {c['foreign_facts']} foreign facts, by tier {c['by_tier']}")
        return
    if args.write_baseline:
        rec = write_baseline(baseline_path=args.baseline)
        print(f"wrote baseline {rec['path']}: {rec['dirs']:,} entity dirs at {rec['captured_at']}")
        return
    s = snapshot(days=args.days, baseline_path=args.baseline)
    if args.json:
        print(json.dumps(s, indent=2))
        return
    c, n, r = s["contamination"], s["near_duplicates"], s["regrowth"]
    print("Knowledge-graph hygiene")
    print(f"  contamination   {c['dirs']:>6} dirs hold {c['foreign_facts']} facts about another entity  {c['by_tier']}")
    print(f"  near-duplicates {n['clusters']:>6} clusters over {n['dirs']} dirs  {n['by_tier']}")
    print(f"  regrowth        {describe(r)}  {r['by_tier']}")
    pv = s.get("provenance") or {}
    if "facts" in pv:
        print(f"  provenance      {pv['both_pct']:>6}% of {pv['facts']:,} facts carry both created_at and source_doc")


if __name__ == "__main__":
    main()
