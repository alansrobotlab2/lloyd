#!/usr/bin/env python3
"""Link stranded entities to the registered entity name inside their own name — #1019.

Dry-run by default. Nothing is written without --apply, and --apply is refused
while `knowledge_graph.write_enabled` is false — the flag `kg_rebuild.py` sets
for the duration of a rebuild freeze and restores at the end. That refusal is
what `writes_disabled_by_rebuild` below is for; the dry run is never refused.

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

WHAT THE #1618 RULING SETTLED, AND WHAT THIS TOOL STILL DOES NOT DECIDE. The
bare↔role-noun shape (`Browser` ↔ `Browser Tool`) is refused by the #320 merge
policy, and #1618 ruled on 2026-09-29 whether linking it is wanted: the SHAPE is
admitted, and the generic-role-noun CLASS is refused by a property measured off
the store — never by a list of names, which is why nothing in this file knows the
word `tool`. `find_proposals` therefore admits a candidate only when its TARGET
passes the bound: fan-out (other registered entity names that embed it, under
this module's own `MIN_TARGET_LEN` and `_boundary_re`) at most
`TARGET_FAN_OUT_MAX` = 20, active degree at least `TARGET_DEGREE_MIN` = 1, and
either at least `TARGET_FACTS_MIN` = 5 active facts or fan-out at most
`TARGET_FAN_OUT_TRUSTED` = 2. The two refusals print their own denominators:
`refused: dead target` is the degree condition (a target with no active edge is a
second stranded node, not a hub — `tool`, `cache` and `automod` on the
2026-09-29 store, each at fan-out 100, 24 and 13 with zero edges), and
`refused: generic target` is the fan-out condition (`VLA` 79, `Inference` 43,
`Prompt` 32, and `Guard`, which the facts/fan-out clause is there for: fan 5 with
only 3 facts, so its name distinguishes nothing). Those thresholds were set by
that measurement, which on the same store took the 18 embedding candidates at
`--min-facts 10` to 11 admitted and 7 refused. The bound is not a verdict on any
one name: `Genesis` at fan-out 1, 12 facts and degree 10 is admitted, and the
`Tactile Genesis → Genesis` split it therefore still causes is #1019's cost to
carry, not this bound's to refuse. What this tool still does not decide is the
merge question #320 poses: the edge written here is connectivity and never an
alias, and #1019's post-landing clause still puts a person in front of the first
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

# The #1618 target-side bound, ruled 2026-09-29 and applied by `find_proposals`.
# The bare↔role-noun SHAPE is admitted; the generic-role-noun CLASS is refused by
# a property measured off the store, so no list of names is kept by hand and
# nothing in this file knows the word `tool`. Thresholds are the ones set by the
# 2026-09-29 measurement (18 embedding candidates at `--min-facts 10`: 11
# admitted, 7 refused), documented in the module docstring.
TARGET_FAN_OUT_MAX = 20       # above this fan-out, the name is a suffix, not a thing
TARGET_DEGREE_MIN = 1         # a target with no active edge links nothing to anything
TARGET_FACTS_MIN = 5          # generic-shaped, but written about enough to keep
TARGET_FAN_OUT_TRUSTED = 2    # at or below this fan-out, the name distinguishes itself


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


# ── the #1618 target-side bound ──────────────────────────────────────────────

def target_fan_out(target: str, names) -> int:
    """How many OTHER registered entity names embed `target`, boundary-clean.

    The generic-role-noun test, and the reason no word list is needed: `tool` is
    generic not because someone wrote it down as one but because 100 registered
    names contain it, so a link to `tool` says something about almost every
    entity and about nothing in particular. A name shorter than `MIN_TARGET_LEN`
    returns 0 rather than a count of the noise it would make — such a target can
    never have been chosen as a target in the first place, since
    `longest_embedded_name` refuses to rank one, and inventing a fan-out for it
    here would let a number print that no matcher ever produced.

    The registry, not an edge query: an edge query would measure half of this
    tool's own output and make the count self-referential — one night's proposals
    would raise the bar for the next.
    """
    tgt = (target or "").strip().lower()
    if len(tgt) < MIN_TARGET_LEN:
        return 0
    rx = _boundary_re(tgt)
    return sum(1 for n in names
               if n and n.strip() != target.strip() and rx.search(n.strip().lower()))


def target_admissibility(*, fan_out: int, degree: int, facts: int) -> str:
    """`admitted`, `refused_dead_target` or `refused_generic_target`.

    The dead check goes FIRST, and the order is not cosmetic: a name can be both
    a dead end and a suffix — `tool` has 100 embedding names and no edges — and
    each such target must land in exactly one row for the counts to partition the
    embedding candidates. It is refused as a dead target because that is the
    cheaper fact to state and the one the operator can act on: a hub the store has
    no edge for is a gap in the graph, not a name that means too much.

    A fan-out above `TARGET_FAN_OUT_MAX` is generic outright. Below it, the
    name may still be a role noun that is simply rarely reused, which is where the
    second half of the clause decides: `Guard` (fan-out 5, 3 facts) is refused
    because nothing written about it distinguishes it, while `Scope Creep`
    (fan-out 1, 4 facts) and `Hook` (fan-out 11, 6 facts) are admitted — the first
    because its fan-out is inside `TARGET_FAN_OUT_TRUSTED`, the second because the
    store has enough to say about it. Reaching the dead row can cost a target
    everything else: `automod` on the 2026-09-29 store has 23 active facts and
    fan-out 13, clears both numeric bars, and is still refused because it holds no
    edge at all. The degree condition is not a proxy for the other two.
    """
    if degree < TARGET_DEGREE_MIN:
        return "refused_dead_target"
    if fan_out > TARGET_FAN_OUT_MAX:
        return "refused_generic_target"
    if facts < TARGET_FACTS_MIN and fan_out > TARGET_FAN_OUT_TRUSTED:
        return "refused_generic_target"
    return "admitted"


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
        "refused_generic_target": 0,
        "refused_dead_target": 0,
        "skipped_no_entity_row": 0,
        "skipped_existing_edge": 0,
    }
    proposed: list[dict] = []
    # The registry to scan for fan-out, and one cache per distinct target: the scan
    # is over ~11.6k names, and a name can be a target of several candidates on one
    # night (`Obsidian` is, on the store this bound was measured on), so without the
    # cache one target is counted once per proposal for no extra information.
    registry = registered
    fan_out_by_target: dict[str, int] = {}
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
        target, substring = hit
        # The #1618 bound, ahead of the two write-path skips: a target this tool
        # must not link to is refused whether or not the write would have worked,
        # so the report distinguishes "we decided no" from "we could not do it".
        if target not in fan_out_by_target:
            fan_out_by_target[target] = target_fan_out(target, registry)
        kind = target_admissibility(
            fan_out=fan_out_by_target[target],
            degree=linked.get(target, 0),
            facts=facts.get(target, 0))
        if kind != "admitted":
            dens[kind] += 1
            continue
        if name.strip().lower() not in have_row:
            dens["skipped_no_entity_row"] += 1
            continue
        if st.edges.active(either=name):
            # Degree is computed from the same table a moment before the write,
            # so this only fires if something wrote an edge mid-pass.
            dens["skipped_existing_edge"] += 1
            continue
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
    cannot drift apart and a test can read the same table the terminal prints.

    The two refusal rows are slices of `embed_a_name` for the same reason
    `proposed` is, and they are NEW KEYS rather than a change to `proposed`
    because `proposed` is what the nightly artefact parser reads
    (`skills/autonomy-data-pipeline/SKILL.md`): a run that refuses seven
    candidates must still report the eleven it proposed under the label the
    pipeline already knows. Every row after `of-which embed a name` is one bucket
    of that set — proposed, refused twice over for the two halves of the #1618
    bound, and skipped twice over for the two write-path reasons — and the
    invariant that they sum to it is pinned in
    `tests/test_link_stranded_entities.py`.
    """
    return [
        ("fact-holding entities", "fact_holding_entities", "", ""),
        ("degree-zero candidates", "degree_zero_candidates",
         "fact_holding_entities", "fact-holding"),
        ("of-which embed a name", "embed_a_name",
         "degree_zero_candidates", "degree-zero candidates"),
        ("proposed edges", "proposed", "embed_a_name", "embedding a name"),
        ("refused: generic target", "refused_generic_target",
         "embed_a_name", "embedding a name"),
        ("refused: dead target", "refused_dead_target",
         "embed_a_name", "embedding a name"),
        # The two write-path skips are slices of the same set: each is a candidate
        # that embedded a name and got no further. Naming the denominator on their
        # rows is what lets the PRINTED table be read as a partition of
        # `embed_a_name`, rather than a list where four rows say whose slices they
        # are and two decline to.
        ("skipped: no entities row", "skipped_no_entity_row",
         "embed_a_name", "embedding a name"),
        ("skipped: edge already live", "skipped_existing_edge",
         "embed_a_name", "embedding a name"),
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


def writes_disabled_by_rebuild() -> bool:
    """True while a knowledge-graph rebuild holds writes off the tree (#1833).

    `kg_rebuild.py:145` flips `knowledge_graph.write_enabled` false and extracts
    into a parallel tree it can rename `facts-quarantine-<ts>` at any moment;
    `agent_mcp/facts.py::_writes_enabled` refuses fact writes in that window, and
    an edge written here is the same hazard with a worse failure mode: a rebuild
    re-derives the graph from fact prose, prose cannot reproduce a name-embedding
    edge, and the pilot's work disappears with no error to read afterwards.

    Read, and failed open, the way every other reader here does — a missing
    config is not a rebuild, and dry-run is the default, so this can only ever
    block an explicit `--apply`. The import is inside the function on purpose:
    this module's two other app imports (`app.kg_store`, `app.paths`) do not pull
    `app.config` in, so a hand run that started before a freeze still reads
    `config.yaml` at the check point, after the multi-second `find_proposals`
    pass, which is the window a freeze has to open in.
    """
    from app.config import CONFIG
    try:
        return not bool(CONFIG.get("knowledge_graph", {}).get("write_enabled", True))
    except Exception:
        return False


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

    # Outside the `if proposed:` branch below, and before the backup: an apply
    # that found nothing must refuse just as loudly as one that found 18 edges,
    # because `applied 0 edges (nothing proposed)` is the exact print a run whose
    # freeze opened mid-pass would leave behind, and it reads as a clean graph.
    if args.apply and writes_disabled_by_rebuild():
        print("\nREFUSED --apply: writes are disabled (config.yaml "
              "knowledge_graph.write_enabled = false), which kg_rebuild.py sets "
              "while a rebuild runs. No edge was written and no backup was taken; "
              "the denominators above are the dry-run reading of this store. "
              "Re-run --apply once the rebuild lands.")
        st.close()
        return 2

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
