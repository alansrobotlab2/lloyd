#!/usr/bin/env python3
"""Regenerate the knowledge graph's rule-derived alias rows (#1486).

The 2026-09-22 deletion took the alias table from 4,028 rows to the 49 the
2026-09-23 rebuild re-seeded, and nothing re-derived the rest (#1399 closed
"stale" because no pre-wipe copy survived). This is the re-derivation for the
part that needs no judgement: surfaces an entity's OWN NAME already spells.

Rules, each stamped into the row's ``origin`` as ``regen:<rule>``:

``paren-acronym``
    ``Causal Agent Replay (CAR)`` → surfaces ``CAR`` and ``Causal Agent Replay``.
``paren-strip``
    ``6D pose (template)`` → ``6D pose``. A year in the parentheses
    (``Ameli et al. (2024)``) is a citation, never stripped: two papers by one
    author would collapse onto one row.
``punct``
    ``Qwen3.8-Flash-Next`` → ``Qwen3.8 Flash Next``: ``-``/``_`` read as a space.

A candidate surface is written only when all of these hold, because an alias
that routes a real name somewhere else is a merge by another name (the
2026-09-03 suffix incident is why ``revert-suffix-merges.py`` exists):

* it is not itself an entity row, and not already an alias row;
* exactly one canonical produced it (a surface two names both yield is dropped
  and reported, never guessed);
* it is not generic (``app.entity_naming._GENERIC_SINGLE``), a bare number, or
  shorter than 2 characters (3 for anything that is not an all-caps acronym).

Semantic aliases — two DIFFERENT names for one thing — are not produced here;
that is a judgement (#1478's djev second judge) and belongs to
``entity-resolution-sweep.py``'s reviewed path.

Writes go only through ``app.kg_store``'s single alias writer, which stamps
``created_at``. Dry-run by default; ``--apply`` writes. The store is whichever
``LLOYD_KG_DB`` / ``app.paths.VAULT_KG_DB`` names — point ``LLOYD_KG_DB`` at a
copy to measure before touching the live one.

Deliberately unscheduled (#4034). No job, autonomy task or unit runs this script,
and that is a decision, not an omission. The evidence:

* the live store holds zero ``regen:`` rows (read 2026-11-10/11: 281 alias rows,
  227 ``sweep``/punct + 54 ``schema``/semantic, against ~17,000 entities) — the
  script has never been applied there;
* on 2026-09-25, on the 86-query gold set (the same 86 the 2026-11-11 re-read
  used), arm E (regen alone, 2,100 rows on a copy) moved no metric, and control G
  (regen + alias surfaces) was identical to arm B (alias surfaces alone) on every
  metric (``eval/measurements/entity-anchoring-2026-09-25.md``);
* the gold-relevance read over the live registry on 2026-11-11: 3,065 planned
  rows, of which 1 is gold-relevant — 1 of 86 gold queries reachable (``TGS RAG``
  → ``TGS-RAG``, whose query already spells the hyphenated name). One query cannot
  move ``entity_hit`` or the anchorless count past its interval. ``--gold-check``
  re-run read-only on 2026-11-26: 3,281 planned rows, still 1 gold-relevant row,
  1 of the 88 queries the gold file holds now.

The punct class is mostly harvested already, by the scheduled
``entity-resolution-sweep.py --apply --tiers CASE,PUNCT`` (``origin="sweep"``).
Re-test the decision with ``--gold-check`` as the registry grows; it opens the store
read-only and writes nothing. At what ``gold_relevant_rows`` a full 09-25-style A/B
earns its cost is Alan's ruling, and arming a recurring job is his too; until then
``--apply`` is for a copy (``LLOYD_KG_DB``), never the live store.

Usage:
  regenerate_aliases.py                       # dry-run, report to stdout
  regenerate_aliases.py --report out.json     # dry-run, full report
  regenerate_aliases.py --gold-check          # read-only: planned vs gold-relevant rows
  LLOYD_KG_DB=/path/copy.sqlite regenerate_aliases.py --apply --report out.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from app.entity_naming import _GENERIC_SINGLE  # noqa: E402

ORIGIN_PREFIX = "regen"
RULES = ("paren-acronym", "paren-strip", "punct")

_PAREN_RE = re.compile(r"^(?P<head>.+?)\s*\((?P<inner>[^()]+)\)\s*(?P<tail>.*)$")
_YEAR_RE = re.compile(r"^\s*(19|20)\d\d[a-z]?\s*$")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9&.\-]{1,9}s?$")
_NUMERIC_RE = re.compile(r"^#?[\d.\-_ ]+$")


def _acronym_matches(head: str, acro: str) -> bool:
    """`CAR` for `Causal Agent Replay`: initials of the head's words, in order.

    Loose on purpose (a letter may be skipped for `and`/`of`, a trailing `s`
    is plural), strict where it matters: every acronym letter must be the first
    letter of a distinct head word, left to right.
    """
    letters = [c for c in acro.lower() if c.isalpha()]
    if letters and letters[-1] == "s" and len(letters) > 2:
        letters = letters[:-1]
    words = [w for w in re.split(r"[\s\-_/]+", head.lower()) if w]
    i = 0
    for w in words:
        if i < len(letters) and w[0] == letters[i]:
            i += 1
    return i == len(letters) and len(letters) >= 2


def candidates_for(name: str) -> list[tuple[str, str]]:
    """(surface, rule) pairs the rules derive from one entity name."""
    out: list[tuple[str, str]] = []
    m = _PAREN_RE.match(name)
    if m and not m.group("tail"):
        head, inner = m.group("head").strip(), m.group("inner").strip()
        if not _YEAR_RE.match(inner):
            if _ACRONYM_RE.match(inner) and _acronym_matches(head, inner):
                out.append((inner, "paren-acronym"))
            out.append((head, "paren-strip"))
    if re.search(r"[-_]", name) and " " in name.replace("-", " ").replace("_", " ").strip():
        spaced = re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", name)).strip()
        if spaced and spaced != name:
            out.append((spaced, "punct"))
    return out


def _admissible(surface: str) -> bool:
    s = surface.strip()
    if _NUMERIC_RE.match(s):
        return False
    if s.lower() in _GENERIC_SINGLE:
        return False
    if _ACRONYM_RE.match(s):
        return len(s) >= 2
    return len(s) >= 3


def plan(entity_names: list[str], existing_alias_lc: set[str]) -> dict:
    """Pure: which alias rows the rules would write over this registry."""
    entity_lc = {n.lower() for n in entity_names}
    by_surface: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for name in entity_names:
        for surface, rule in candidates_for(name):
            by_surface[surface.lower()].add((surface, name, rule))
    rows, collisions, skipped = [], [], defaultdict(int)
    for lc, cands in sorted(by_surface.items()):
        canonicals = {c for _s, c, _r in cands}
        if lc in entity_lc:
            skipped["is_entity"] += 1
            continue
        if lc in existing_alias_lc:
            skipped["already_alias"] += 1
            continue
        if len(canonicals) > 1:
            collisions.append({"surface": lc, "canonicals": sorted(canonicals)})
            continue
        surface, canonical, rule = sorted(cands)[0]
        if not _admissible(surface):
            skipped["inadmissible"] += 1
            continue
        rows.append({"surface": surface, "canonical": canonical, "rule": rule})
    return {"rows": rows, "collisions": collisions, "skipped": dict(skipped)}


def _kind_for(rule: str) -> str:
    return {"punct": "punct", "paren-strip": "suffix", "paren-acronym": "semantic"}[rule]


def run(apply: bool, report: Path | None = None) -> dict:
    from app.kg_store import store
    st = store()
    before = st.aliases.count()
    existing = set(st.aliases.all_lower()) - {n.lower() for n in st.entities.all()}
    p = plan(st.entities.all(), existing)
    written = 0
    if apply:
        for r in p["rows"]:
            st.aliases.set(r["surface"], r["canonical"], kind=_kind_for(r["rule"]),
                           origin=f"{ORIGIN_PREFIX}:{r['rule']}",
                           report_path=str(report) if report else None)
            written += 1
    out = {
        "store": str(st.path),
        "applied": apply,
        "aliases_before": before,
        "aliases_after": st.aliases.count(),
        "planned": len(p["rows"]),
        "written": written,
        "by_rule": {rule: sum(1 for r in p["rows"] if r["rule"] == rule) for rule in RULES},
        "collisions": len(p["collisions"]),
        "skipped": p["skipped"],
    }
    if report:
        try:
            from _invocation import invocation_ledger
            ledger = invocation_ledger()
        except Exception:  # noqa: BLE001 — the ledger is provenance, not the job
            ledger = None
        Path(report).write_text(json.dumps({**out, "invocation": ledger, "rows": p["rows"],
                                            "collision_rows": p["collisions"]},
                                           indent=1, ensure_ascii=False))
    return out


#: The gold set `eval/run_eval.py` scores nightly.
DEFAULT_GOLD = HERE.parent.parent / "eval" / "vault_recall_queries.yaml"


def _spaced(text: str) -> str:
    """Lowercase, `-`/`_` and every other non-alphanumeric read as a space."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(text).lower().replace("_", " ")).split())


def _eval_norm(text: str) -> str:
    """`eval/run_eval.py::_norm`: lowercase, `-` and `_` equivalent."""
    return str(text or "").lower().replace("-", "_")


def _satisfies(canonical: str, gold: list[str]) -> bool:
    """The eval's lenient entity rule: a gold name and the canonical contain one
    another (either way), after `_eval_norm`."""
    c = _eval_norm(canonical)
    return any(g and (g in c or c in g) for g in (_eval_norm(x) for x in gold))


def gold_check(planned_rows: list[dict], queries: list[dict]) -> dict:
    """Pure: which planned rows could change what a gold query anchors on.

    A row is gold-relevant when its surface, spaced, appears as a whole token run
    inside a gold query's spaced text AND its canonical satisfies that same query's
    `expect_entities`. A surface that sits in a query whose gold it cannot satisfy
    is counted as a `surface_hit` only — it would seed something the query is not
    scored on.
    """
    relevant: list[dict] = []
    reachable: set[str] = set()
    surface_hits = 0
    spaced_q = [(q, f" {_spaced(q.get('query', ''))} ") for q in queries]
    for row in planned_rows:
        needle = _spaced(row["surface"])
        if not needle:
            continue
        hit_ids = []
        for q, text in spaced_q:
            if f" {needle} " not in text:
                continue
            surface_hits += 1
            if _satisfies(row["canonical"], q.get("expect_entities") or []):
                hit_ids.append(str(q.get("id")))
        if hit_ids:
            relevant.append({**row, "queries": hit_ids})
            reachable.update(hit_ids)
    return {
        "planned_rows": len(planned_rows),
        "gold_queries": len(queries),
        "surface_hits": surface_hits,
        "gold_relevant_rows": len(relevant),
        "gold_queries_reachable": len(reachable),
        "relevant": relevant,
    }


def run_gold_check(gold: Path = DEFAULT_GOLD, db: Path | None = None) -> dict:
    """`--gold-check`: the store through a `?mode=ro` handle, the gold file read,
    nothing written anywhere."""
    import yaml
    from app.kg_store import read_only_registry
    names, alias_lc = read_only_registry(db)
    existing = alias_lc - {n.lower() for n in names}
    queries = (yaml.safe_load(Path(gold).read_text(encoding="utf-8")) or {}).get("queries") or []
    return gold_check(plan(names, existing)["rows"], queries)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the rows (default: dry-run)")
    ap.add_argument("--report", type=Path, help="write the full plan/report JSON here")
    ap.add_argument("--gold-check", action="store_true",
                    help="read-only: print planned_rows and gold_relevant_rows; writes nothing")
    ap.add_argument("--gold", type=Path, default=DEFAULT_GOLD,
                    help="gold query file for --gold-check (default: %(default)s)")
    args = ap.parse_args(argv)
    if args.gold_check:
        if args.apply or args.report:
            ap.error("--gold-check writes nothing: it takes neither --apply nor --report")
        out = run_gold_check(args.gold)
        for key in ("planned_rows", "gold_queries", "surface_hits",
                    "gold_relevant_rows", "gold_queries_reachable"):
            print(f"{key}: {out[key]}")
        for row in out["relevant"]:
            print(f"relevant: {row['surface']!r} -> {row['canonical']!r} "
                  f"({row['rule']}) in {', '.join(row['queries'])}")
        return 0
    print(json.dumps(run(args.apply, args.report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
