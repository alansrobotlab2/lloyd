#!/usr/bin/env python3
"""Link stranded entities to the registered entity name inside their own name — #1019.

Dry-run by default. Nothing is written without --apply.

WHY THIS EXISTS. An entity gets an edge only two ways, and both can miss it
entirely: the extractor links entities *named inside a fact's prose*
(`scripts/memory/next-gen-memory/fact_extractor.py:551`,
`_known_entities(text, 12)`), so a self-referential fact — "The MRR for
#363 TGS-RAG Implementation improved from 0.284 to 0.500" — names no second
entity and the row is born at degree zero; and the backfill
`scripts/memory/seed_relationship_edges.py:205` only scans entity dirs that
carry a `<Entity>-relationship.md`, which 96.3% of the stranded mass measured
2026-09-19 do not have. A degree-zero row is invisible to every graph leg of
retrieval: `agent_mcp/retrieval.py` expands neighbours from adjacency only, so
no seed match, no neighbour weighting and no re-ranking knob can reach it.

THE ONE MECHANICAL EVIDENCE THAT NEEDS NO PROSE. Such an entity's own name
often contains another *registered* entity name — `#363 TGS-RAG Implementation`
contains `TGS-RAG` — which says, without reading a word of fact text, that the
row is about the thing that name denotes. That is the rule this tool applies:
for every fact-holding entity sitting at zero active edges, propose exactly one
active `related_to` edge to the LONGEST other registered entity name embedded
case-insensitively in its own name. It proposes nothing for an entity whose
name embeds no other registered name. A registered name equal to the entity's
own name modulo case is not "another" name — a link to one is a self-loop
wearing a hat.

WHAT "EMBEDDED" MEANS, AND WHAT IT COSTS. Boundary-clean and case-insensitive:
the occurrence may touch space or punctuation — `TGS-RAG` inside
`#363 TGS-RAG Implementation` sits against `#` and a space, so a `\b`-style rule
is not enough and the code uses look-around on letters and digits — but it may
not sit inside a longer word. The first dry run over a copy of the live store
used plain substring and proposed 1,694 edges including
`Geometric Deep Learning → METR` and `Harrison Chase → HASE`: `metr` is inside
`geometric`, `hase` inside `Chase`. Those are not relations, and the item's own
verification probe applies precisely this boundary rule, so a matcher wider than
the check would write edges the check then refuses to count. The cost is named:
`CameraCfg`, stranded with 49 facts, stays stranded, because `Cfg` occurs in it
only inside the word.

WHY `origin="manual"`. `scripts/memory/kg_rebuild.py` exports for re-derivation
keeps an edge when its origin is in `kg_store.CARRY_EDGE_ORIGINS` (or its
provenance is in `CARRY_PROVENANCE`) — extraction cannot reproduce an edge no
fact prose names, which is exactly this case, so the pair written here
(`origin="manual"`, `provenance="INFERRED"`) survives a rebuild twice over.
The value is imported from `app.kg_store`, not restated: `CARRY_EDGE_ORIGINS`
used to be a literal in the export filter and #1151's rule is that a value
several readers share lives with the writer.

WHAT THIS TOOL DELIBERATELY DOES NOT DECIDE. The bare↔role-noun shape
(`Browser` ↔ `Browser Tool`) is refused by the #320 merge policy, and whether
linking it is wanted is a scope call no test can settle, so it is NOT suppressed
here — #1019's post-landing clause puts it in front of a person before the first
live `--apply`. The confidence is the store default (0.5) on purpose: the
contract is connectivity, not retrieval rank, and the triage arithmetic measured
that no `related_to` confidence reaches the top-5 neighbour cutoff anyway
(0.2284 against a 5th-place 0.7234 on 2026-09-19), so raising it would buy
nothing and would misstate how much the name-sharing actually proves.

Usage:
  python scripts/memory/link_stranded_entities.py                  # dry-run
  python scripts/memory/link_stranded_entities.py --min-facts 10   # the probe's floor
  python scripts/memory/link_stranded_entities.py --sample 20      # show pairs
  python scripts/memory/link_stranded_entities.py --apply
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.kg_store import CARRY_EDGE_ORIGINS, KGStore  # noqa: E402
from app.paths import VAULT_KG_DB  # noqa: E402

# The edge shape. `manual` must stay inside CARRY_EDGE_ORIGINS or a rebuild
# re-derivation erases every edge this tool writes; the import above makes the
# coupling a real dependency instead of a comment, and the assertion below makes
# it fail at import rather than at the next rebuild.
EDGE_TYPE = "related_to"
EDGE_ORIGIN = CARRY_EDGE_ORIGINS[1] if "manual" in CARRY_EDGE_ORIGINS else "manual"
EDGE_PROVENANCE = "INFERRED"
EDGE_CONFIDENCE = 0.5
EVIDENCE_TEMPLATE = "name-embedding: {substring!r} is a registered entity name inside {source!r}"

assert EDGE_ORIGIN in CARRY_EDGE_ORIGINS, (
    f"{EDGE_ORIGIN!r} is no longer carried across a rebuild re-derivation")

MIN_TARGET_LEN = 3          # 1- and 2-char names match almost anything, letter by letter
SEED = 1019


@lru_cache(maxsize=8192)
def _boundary_re(needle_lc: str) -> re.Pattern:
    """The case-folded needle, flanked by anything but a letter or a digit.

    Hyphen, `#` and space are boundaries — `TGS-RAG` in
    `#363 TGS-RAG Implementation` needs them to be — while a letter or a digit is
    not, which is what keeps `METR` out of `Geometric` and `HASE` out of `Chase`.
    This is the same look-around the item's verification probe applies, so the
    writer and the check agree on what "embedded" means.
    """
    return re.compile(r"(?<![0-9a-z])" + re.escape(needle_lc) + r"(?![0-9a-z])")


# ── the matcher ──────────────────────────────────────────────────────────────

def build_target_index(names) -> dict[str, list[str]]:
    """first character → registered names beginning with it.

    A prefilter, not the test: if `X` is embedded in `S` then `X`'s first
    character is a character of `S`, so a name can only be embedded in a source
    whose character set contains its first character. Grouping that way and then
    doing the real substring test on the survivors keeps the pass over ~11.6k
    registered names and ~2.5k candidates to seconds instead of minutes.

    The prefilter is deliberately weaker than a word-boundary index would be. The
    rule #1019 states is literal — "embedded case-insensitively in its own name"
    — and a token index would silently drop the joined-word shape that rule
    reaches: `Cfg` is embedded in `CameraCfg` at no word boundary at all, and
    `CameraCfg` is one of the degree-zero rows the item names.
    """
    idx: dict[str, list[str]] = {}
    for n in names:
        lc = (n or "").strip().lower()
        if lc:
            idx.setdefault(lc[0], []).append(n.strip())
    return idx


def longest_embedded_name(name: str, index: dict[str, list[str]]) -> tuple[str, str] | None:
    """(registered name, case-folded shared substring) for the longest other
    registered entity name embedded in `name`, or None.

    Longest wins so `#363 TGS-RAG Implementation` resolves to `TGS-RAG` and not
    to the `RAG` inside it; equal lengths break alphabetically so two runs over
    one store propose the same edge. A source shorter than the candidate is
    skipped: boundary containment already implies it, and the guard above makes
    that dependence explicit rather than incidental.
    """
    src_lc = (name or "").strip().lower()
    if not src_lc:
        return None
    cands: set[str] = set()
    for ch in set(src_lc):
        cands.update(index.get(ch, ()))
    best: tuple[str, str] | None = None
    for cand in cands:
        cand_lc = cand.strip().lower()
        if cand_lc == src_lc or len(cand_lc) < MIN_TARGET_LEN or len(cand_lc) >= len(src_lc):
            continue
        if not _boundary_re(cand_lc).search(src_lc):
            continue
        if best is None or (-len(cand_lc), cand_lc) < (-len(best[0]), best[1]):
            best = (cand, cand_lc)
    return best


# ── store reads ──────────────────────────────────────────────────────────────

def active_fact_counts(st: KGStore) -> dict[str, int]:
    """entity label → active facts, by exact label.

    Deliberately raw SQL over `facts_idx` rather than
    `facts_idx.entity_fact_counts()`, which folds case: this tool's whole job is
    distinguishing one label from another, and the probe it is verified against
    groups on the exact label.
    """
    return {r["entity"]: int(r["c"]) for r in st._query(
        "SELECT entity, COUNT(*) c FROM facts_idx "
        "WHERE expired_at IS NULL OR expired_at='' GROUP BY entity")}


def find_proposals(st: KGStore, *, min_facts: int = 0,
                   limit: int | None = None) -> tuple[list[dict], dict[str, int]]:
    """(proposed edges, denominators) for the stranded entities this rule reaches.

    Candidates are fact-holding labels at zero active edges. A label with no
    `entities` row is counted and skipped rather than linked: writing an edge
    onto an unregistered node would make the graph say a row exists that does
    not, and the denominator is what tells the next reader that mass exists.
    """
    registered = st.entities.all()
    index = build_target_index(registered)
    have_row = {n.strip().lower() for n in registered}
    facts = active_fact_counts(st)
    linked = st.edges.degree()

    dens: dict[str, int] = {
        "fact_holding_entities": len(facts),
        "degree_zero_candidates": 0,
        "embed_a_name": 0,
        "proposed": 0,
        "skipped_no_entity_row": 0,
        "skipped_existing_edge": 0,
    }
    proposed: list[dict] = []
    for name, n_facts in sorted(facts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n_facts < min_facts:
            continue
        if linked.get(name, 0) > 0:
            continue
        dens["degree_zero_candidates"] += 1
        hit = longest_embedded_name(name, index)
        if hit is None:
            continue
        dens["embed_a_name"] += 1
        if name.strip().lower() not in have_row:
            dens["skipped_no_entity_row"] += 1
            continue
        if st.edges.active(either=name):
            # Degree is computed from the same table a moment before the write,
            # so this only fires if something wrote an edge mid-pass.
            dens["skipped_existing_edge"] += 1
            continue
        target, substring = hit
        proposed.append({
            "source": name,
            "target": target,
            "type": EDGE_TYPE,
            "confidence": EDGE_CONFIDENCE,
            "provenance": EDGE_PROVENANCE,
            "origin": EDGE_ORIGIN,
            "evidence": EVIDENCE_TEMPLATE.format(substring=substring, source=name),
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        dens["proposed"] = len(proposed)
        if limit is not None and len(proposed) >= limit:
            break
    return proposed, dens


# ── reporting ────────────────────────────────────────────────────────────────

def report_lines(dens: dict[str, int]) -> list[tuple[str, str, str, str]]:
    """(label, denominator key, denominator label, that denominator's label) —
    the report's shape in one place, so the counts and the sets they are slices of
    cannot drift apart and a test can read the same table the terminal prints."""
    return [
        ("fact-holding entities", "fact_holding_entities", "", ""),
        ("degree-zero candidates", "degree_zero_candidates",
         "fact_holding_entities", "fact-holding"),
        ("of-which embed a name", "embed_a_name",
         "degree_zero_candidates", "degree-zero candidates"),
        ("proposed edges", "proposed", "embed_a_name", "embedding a name"),
        ("skipped: no entities row", "skipped_no_entity_row", "", ""),
        ("skipped: edge already live", "skipped_existing_edge", "", ""),
    ]


def print_denominators(dens: dict[str, int], mode: str) -> None:
    """Every count this tool acts on, printed beside its denominator.

    A bare `0 proposed` is the reading error this block exists to prevent: 0 is
    also the output of an empty candidate set, of a matcher that never fired, and
    of a store that failed to load. Each line therefore carries the number it is
    a slice OF.
    """
    print(f"Stranded-entity linker — #1019 ({mode})")
    for label, key, of, of_label in report_lines(dens):
        tail = f"  (of {dens[of]:,} {of_label})" if of else ""
        print(f"  {label:26s} {dens[key]:>8,}{tail}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Link degree-zero entities to the registered entity name in their own name (#1019)")
    ap.add_argument("--apply", action="store_true", help="write the proposed edges")
    ap.add_argument("--min-facts", type=int, default=0,
                    help="only act on entities with at least N active facts "
                         "(0 = every fact-holding entity; 10 = the floor #1019's "
                         "probe enumerates). Never a substitute for reading the "
                         "denominators.")
    ap.add_argument("--limit", type=int, help="stop after N proposals (pilot runs)")
    ap.add_argument("--sample", type=int, default=25, help="example pairs to print")
    ap.add_argument("--out", type=Path, help="write proposed edges as JSONL")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the store backup on --apply (tests only)")
    ap.add_argument("--db", type=Path, default=VAULT_KG_DB, help="KG store to act on")
    args = ap.parse_args(argv)

    st = KGStore(args.db)
    proposed, dens = find_proposals(st, min_facts=args.min_facts, limit=args.limit)
    print_denominators(dens, "APPLYING" if args.apply else "dry-run")

    if proposed and args.sample:
        shown = min(args.sample, len(proposed))
        print(f"\n  --- {shown} of {len(proposed):,} proposed edges ---")
        for e in random.Random(SEED).sample(proposed, shown):
            print(f"    {e['source']}  →  {e['target']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(e) + "\n" for e in proposed))
        print(f"\nwrote proposals → {args.out}", file=sys.stderr)

    if args.apply:
        if proposed:
            backup = None
            if not args.no_backup:
                ts = datetime.now().strftime("%Y%m%dT%H%M%SZ")
                backup = st.backup(args.db.parent / "store-backups" / f"kg-stranded-{ts}.sqlite")
            with st.transaction():
                for e in proposed:
                    st.edges.add(e, origin=EDGE_ORIGIN)
            print(f"\napplied {len(proposed):,} edges"
                  + (f"; backup → {backup}" if backup else "; no backup (--no-backup)"))
        else:
            print("\napplied 0 edges (nothing proposed)")
    else:
        print("\n(dry-run — pass --apply to write)")

    st.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
