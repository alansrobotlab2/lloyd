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

Usage:
  regenerate_aliases.py                       # dry-run, report to stdout
  regenerate_aliases.py --report out.json     # dry-run, full report
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="write the rows (default: dry-run)")
    ap.add_argument("--report", type=Path, help="write the full plan/report JSON here")
    args = ap.parse_args(argv)
    print(json.dumps(run(args.apply, args.report), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
