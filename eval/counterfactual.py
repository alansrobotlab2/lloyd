#!/usr/bin/env python3
"""Counterfactual-perturbation rater for the retrieval eval (#541).

One perturbed variant per nightly query: flip exactly one named constraint, run
the variant through the identical retrieval path, and score two independent
directions.

  moved   the changed constraint moved what the retriever pulled
  pinned  the constraints that did not change pulled the same rows as before

Both directions are the point. A moved-only rater rewards an extractor that
churns everything on every edit — and it cannot see an *incorrectly unchanged*
result, which is the specific signature of a retriever keying on surface
bag-of-words instead of the named thing. Since entity_hit_rate sits near 0.5
while doc_hit_rate sits near 0.95, that is the failure this measures.

Two design constraints, both from the item's own risks:

The records are **committed, never re-derived at eval time**. The sibling pairs
come from the graph alias table (``sibling_source: kg_alias_table``), but they
were drawn once, by hand, from pairs already known to be normalization
artifacts, and then frozen into ``counterfactual_perturbations.yaml``. If the
live table picked the pairs every night the measurement would change underneath
the trend line — the defect the nightly compare step already guards against for
the query set. ``verify_siblings`` is the audit against the live table, run by
hand, not by the nightly path.

The metrics are **entity/fact-level only**. The doc corpus is the same vault in
both arms, so any doc-level comparison reads as spurious success (risk 3). The
retrieved projection is built from fact/graph entity attributions and fact text;
``documents`` is never consulted.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
RECORD_PATH = HERE / "counterfactual_perturbations.yaml"

#: The fixed taxonomy. One axis per query, decided by the plan below.
AXES = ("entity", "date", "qualifier", "artifact")

ENTITY_AXIS = "entity"

#: axis_changed -> what the perturbation is allowed to name as a new constraint.
AXES_REQUIRE_MOVE_TARGET = {ENTITY_AXIS}

# ── the axis plan ────────────────────────────────────────────────────────────
# Hand-authored and reviewed, one entry per query id. Sibling values are
# canonicals read out of the graph alias table (surface -> canonical pairs that
# already coexist there, i.e. known normalization artifacts), never invented:
# an unfair sibling manufactures false failures, which is risk 1 in #541.
#
#   expected_to_move   entity names the variant must pull that the original did
#                      not (entity axis only — the other axes name no new row)
#   expected_pinned    constraints the query still names, which must pull the
#                      same rows in both arms. Anything not named here is not
#                      scored, so a query with one constraint contributes to
#                      moved_rate only — a vacuous pin would inflate pinned_rate
# PLAN[id] = (axis, old_value, new_value, expected_to_move, expected_pinned)
PLAN: dict[str, tuple[str, str, str, list[str], list[str]]] = {
    # entity swaps — the axis whose failure mode is the direct evidence for or
    # against #537's typed-identity-key premise.
    "backlog-363": ("entity", "backlog item 363", "Backlog Item #313",
                    ["Backlog Item #313"], []),
    "entity-resolution-sweep": ("entity", "entity resolution sweep",
                               "Semantic Entity Resolution",
                               ["Semantic Entity Resolution"], []),
    "inner-voice": ("entity", "inner voice", "Voice Mode", ["Voice Mode"], []),
    "vault-recall": ("entity", "vault_recall", "Vault Index", ["Vault Index"], []),
    "qmd": ("entity", "QMD", "QMD Search", ["QMD Search"], []),
    "kg-maintenance-tasks": ("entity", "knowledge graph", "Entity Graph",
                            ["Entity Graph"], ["autonomy"]),
    "lloyd-vllm-rel": ("entity", "vLLM", "TensorRT-LLM",
                       ["TensorRT-LLM"], ["lloyd"]),
    # An alias-table normalization pair: 'autonomy pipeline' and 'Data Pipeline'
    # are both live surfaces for Autonomy Data Pipeline. If retrieval cannot tell
    # the two surfaces apart, the seed set does not move — and that is the
    # identity-keying result, not a broken perturbation.
    "autonomy-pipeline": ("entity", "autonomy pipeline", "Data Pipeline",
                          ["Data Pipeline"], ["autonomy"]),
    # artifact-type changes: "the daily note" -> "the knowledge note" family
    "harness-tools": ("artifact", "tool calls", "tool results", [], ["agent harness"]),
    "nightly-reflection": ("artifact", "skills", "autonomy tasks", [],
                           ["nightly reflection"]),
    "relationships-location": ("artifact", "index", "backup", [], ["relationships"]),
    "robotics-projects": ("artifact", "projects", "research notes", [], ["robotics"]),
    # qualifier negation / replacement
    "qwen38-local-serving": ("qualifier", "24GB", "48GB", [], ["Qwen3.8"]),
    "tgs-rag-state": ("qualifier", "current", "original", [], ["TGS-RAG"]),
    "graph-quality": ("qualifier", "noise", "precision", [], ["graph"]),
    "classifier-v4": ("qualifier", "upgrade", "keep", [], ["mention classifier"]),
    "godnode-threshold": ("qualifier", "why does it exist", "what does it cost",
                          [], ["FACT_GODNODE_THRESHOLD"]),
    "memory-persistence": ("qualifier", "across sessions", "within one session",
                           [], ["Lloyd"]),
    "backlog-overview": ("qualifier", "interesting", "stale", [], ["Backlog"]),
    # date swap
    "this-week-autonomy": ("date", "this week", "last quarter", [], ["autonomy"]),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _match(value: str, candidates: set[str]) -> bool:
    """Presence test against a normalized retrieved-entity set.

    Substring in either direction: an expected 'entity graph' is satisfied by a
    retrieved row attributed to 'Entity Graph (KG)', which is what the store
    actually emits for some canonicals.
    """
    needle = _norm(value)
    if not needle:
        return False
    return any(needle in c or c in needle for c in candidates if c)


# ── generation (pure: no store, no network, no LLM) ──────────────────────────

def apply_perturbation(query: str, record: dict) -> str:
    """Swap the one named constraint. Case-insensitive, first occurrence only.

    Replacing only the first occurrence is what keeps this a one-axis change: a
    query that named the constraint twice would otherwise become two edits.
    """
    old, new = record["old_value"], record["new_value"]
    idx = query.lower().find(old.lower())
    if idx < 0:
        raise ValueError(f"{record.get('id', '?')}: {old!r} not in the query")
    return query[:idx] + new + query[idx + len(old):]


def build_perturbations(specs: list[dict]) -> list[dict]:
    """The committed records, regenerated deterministically from the plan.

    Raises rather than silently skipping: if the query set gains an id with no
    plan entry, or a plan entry's old_value stops matching its query, the rater
    would quietly start scoring fewer queries than the trend line claims.
    """
    out: list[dict] = []
    for spec in specs:
        qid = spec["id"]
        if qid not in PLAN:
            raise ValueError(f"{qid}: no perturbation plan entry")
        axis, old, new, to_move, pinned = PLAN[qid]
        if axis not in AXES:
            raise ValueError(f"{qid}: axis {axis!r} outside the taxonomy")
        query = spec["query"]
        if old.lower() not in query.lower():
            raise ValueError(f"{qid}: old_value {old!r} not in its query")
        rec = {
            "id": qid,
            "axis_changed": axis,
            "old_value": old,
            "new_value": new,
            "perturbed_query": apply_perturbation(query, {"old_value": old,
                                                          "new_value": new,
                                                          "id": qid}),
            "expected_to_move": list(to_move),
            "expected_pinned": list(pinned),
        }
        if axis in AXES_REQUIRE_MOVE_TARGET:
            if not to_move:
                raise ValueError(f"{qid}: {axis} axis names no move target")
            rec["sibling_source"] = "kg_alias_table"
        out.append(rec)
    planned = set(PLAN) - {s["id"] for s in specs}
    if planned:
        raise ValueError(f"plan entries with no query: {sorted(planned)}")
    return out


def emit_records(records: list[dict], path: Path = RECORD_PATH) -> None:
    blob = {
        "_comment": ("Deterministic counterfactual perturbations for the "
                     "retrieval eval (#541). Regenerate with "
                     "`python eval/counterfactual.py --write`; never re-derived "
                     "from live graph data at eval time. The pinned corpus and "
                     "vault_recall_queries.yaml ground truth are not read by "
                     "the generator."),
        "perturbations": sorted(records, key=lambda r: r["id"]),
    }
    path.write_text(yaml.safe_dump(blob, sort_keys=False, allow_unicode=True,
                                   default_flow_style=False))


def load_records(path: Path = RECORD_PATH) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — the perturbation records are committed and "
            "the rater cannot score without them")
    blob = yaml.safe_load(path.read_text()) or {}
    return {r["id"]: r for r in blob.get("perturbations") or []}


def verify_siblings(records: list[dict], resolve) -> list[dict]:
    """Audit the sibling chosen for every entity-axis swap (risk 1).

    ``resolve`` maps a name to its canonical, or None. Two things make a swap
    unfair, and only the swapped-in value is held to the table:

      new_value resolves to nothing  — an invented entity cannot be expected to
        move a seed set, so a failure there is the perturbation's fault, not the
        retriever's;
      old and new resolve to the SAME canonical — two surfaces of one entity.
        Asking retrieval to distinguish them asks it to be wrong.

    old_value resolving to nothing is fine and expected: it is the surface the
    query happens to use ('vLLM' is an entity name that was never registered as
    an alias surface). Run by hand after editing the plan, not from the nightly
    path — the alias table is live data and the records are frozen on purpose.
    """
    bad: list[dict] = []
    for rec in records:
        if rec.get("axis_changed") != ENTITY_AXIS:
            continue
        new_c = resolve(rec["new_value"])
        if not new_c:
            bad.append(rec)
            continue
        old_c = resolve(rec["old_value"])
        if old_c and _norm(old_c) == _norm(new_c):
            bad.append(rec)
    return bad


def alias_resolver(db_path: Path) -> callable:
    """surface -> canonical, or identity when the name IS a canonical.

    The alias table stores variant surfaces, so a canonical that has no variant
    is not present as a surface row — resolving through the table alone reports
    'Knowledge Graph' as unknown. That false negative is what made the first
    version of this audit flag all eight entity swaps as unverified.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    surfaces = {s.lower(): c for s, c in
                con.execute("select surface_lc, canonical from aliases")}
    canonicals = {c.lower() for c in
                  con.execute("select distinct canonical from aliases")}
    con.close()

    def resolve(name: str):
        key = (name or "").strip().lower()
        return surfaces.get(key) or (name if key in canonicals else None)

    return resolve


# ── scoring ──────────────────────────────────────────────────────────────────

def _retrieved(result: dict) -> tuple[set[str], dict[str, set[str]]]:
    """Entity-level projection of one recall result: the attributed entity set,
    and the fact text attributed to each entity.

    Never reads ``documents`` — see the module docstring on risk 3.
    """
    entities: set[str] = set()
    by_entity: dict[str, set[str]] = {}
    for key in ("facts", "graph_expanded_facts"):
        for fact in result.get(key) or []:
            ent = _norm(fact.get("entity", ""))
            if not ent:
                continue
            entities.add(ent)
            by_entity.setdefault(ent, set()).add(_norm(fact.get("text", "")))
    for nb in result.get("graph_neighbors_used") or []:
        ent = _norm(nb.get("entity", ""))
        if ent:
            entities.add(ent)
    return entities, by_entity


def _matched_by(value: str, by_entity: dict[str, set[str]]) -> set[str]:
    """Fact text attributed to every retrieved entity matching `value`."""
    needle = _norm(value)
    out: set[str] = set()
    if not needle:
        return out
    for ent, texts in by_entity.items():
        if needle in ent or ent in needle:
            out |= texts
    return out


def score_pair(record: dict, orig_result: dict, orig_seeds: list[str],
               var_result: dict, var_seeds: list[str]) -> dict:
    """moved / pinned for one query and its perturbed twin."""
    axis = record.get("axis_changed")
    retrieved_o, facts_o = _retrieved(orig_result)
    retrieved_v, facts_v = _retrieved(var_result)

    to_move = [t for t in (record.get("expected_to_move") or [])]
    if to_move:
        # Scored against the rows the variant attributed and the original did
        # not — never against the whole variant output. Tolerance to naming is
        # wanted (the store's canonical for 'Semantic Entity Resolution' is
        # 'semantic-entity-resolution-via-graph-embeddings', and refusing to
        # call that a move manufactures the false failure risk 1 warns about),
        # and restricting to `added` is what keeps that tolerance safe: a row
        # already present in the original arm cannot be credited as movement,
        # so 'QMD' never satisfies a swap to 'QMD Index'.
        added = retrieved_v - retrieved_o
        moved = any(_match(t, added) for t in to_move)
    else:
        moved = retrieved_v != retrieved_o

    pins = record.get("expected_pinned") or []
    if pins:
        pinned = True
        broken: list[str] = []
        for pin in pins:
            was = _match(pin, retrieved_o)
            now = _match(pin, retrieved_v)
            if was != now:
                pinned = False
                broken.append(f"{pin}: present={was}->{now}")
                continue
            if was and now and _matched_by(pin, facts_o) != _matched_by(pin, facts_v):
                pinned = False
                broken.append(f"{pin}: facts churned")
    else:
        pinned = None
        broken = []

    seeds_o = {_norm(s) for s in (orig_seeds or []) if _norm(s)}
    seeds_v = {_norm(s) for s in (var_seeds or []) if _norm(s)}

    return {
        "axis_changed": axis,
        "new_value": record.get("new_value"),
        "perturbed_query": record.get("perturbed_query"),
        "expected_to_move": to_move,
        "expected_pinned": pins,
        "counterfactual_moved": bool(moved),
        "counterfactual_pinned": pinned,
        "pinned_unscored": not pins,
        "pinned_failures": broken,
        # Seeds are extracted from the query text, so a swap the extractor never
        # noticed shows up here first — the label #537 needs.
        "seed_moved": seeds_v != seeds_o,
        "retrieved": sorted(retrieved_o),
        "retrieved_variant": sorted(retrieved_v),
        "retrieved_unchanged": retrieved_v == retrieved_o,
    }


def label_failures(records: list[dict]) -> list[dict]:
    """Cause labels over the per-query counterfactual blocks.

    The label set is the defect taxonomy #541 asks for; `label` is None where
    the query behaved. Ordered by diagnostic value, not by frequency: an entity
    swap the seed extractor did not notice is the cleanest existing evidence
    for or against #537's identity-key premise, so it wins over a generic
    'did not move'.
    """
    out: list[dict] = []
    for rec in records:
        block = rec.get("counterfactual") or {}
        label = None
        if block:
            if (block.get("axis_changed") == ENTITY_AXIS
                    and not block.get("seed_moved")):
                label = "entity_swap_seed_set_unchanged"
            elif block.get("counterfactual_pinned") is False:
                label = "pinned_axis_churned"
            elif not block.get("counterfactual_moved"):
                label = "axis_not_moved"
            elif block.get("counterfactual_pinned") is None:
                label = "nothing_pinned"
        out.append({"id": rec.get("id"), "label": label,
                    "axis_changed": block.get("axis_changed"),
                    "new_value": block.get("new_value"),
                    "pinned_failures": block.get("pinned_failures") or []})
    return out


def identity_keying_evidence(labelled: list[dict]) -> list[str]:
    """The query ids where an entity-name swap left the seed set unchanged."""
    return [row["id"] for row in labelled
            if row.get("label") == "entity_swap_seed_set_unchanged"]


def kg_db_path() -> Path:
    """The store the scored run actually read — the same ``VAULT_KG_DB``
    run_eval records in corpus_provenance, so an audit never runs against a
    different graph than the numbers it is checking."""
    import sys

    sys.path.insert(0, str(HERE.parent))
    from app.paths import VAULT_KG_DB

    return Path(VAULT_KG_DB)


def main(argv: list[str]) -> int:
    specs = yaml.safe_load((HERE / "vault_recall_queries.yaml").read_text())["queries"]
    if "--write" in argv:
        records = build_perturbations(specs)  # validates before writing
        emit_records(records)
        print(f"wrote {RECORD_PATH} ({len(records)} records)")
        return 0
    if "--verify" in argv:
        db = kg_db_path()
        if not db.exists():
            print(f"no alias table at {db} — cannot audit the siblings")
            return 2
        recs = list(load_records().values())
        entity_axis = [r for r in recs if r.get("axis_changed") == ENTITY_AXIS]
        bad = verify_siblings(recs, alias_resolver(db))
        for rec in bad:
            print(f"  UNVERIFIED {rec['id']}: {rec['old_value']!r} -> "
                  f"{rec['new_value']!r} — the swapped-in value is not a "
                  "distinct canonical in the alias table")
        print(f"{len(bad)} of {len(entity_axis)} entity-axis pairs unverified "
              f"against {db}")
        return 1 if bad else 0
    if "--check" in argv:
        built = {r["id"]: r for r in build_perturbations(specs)}
        committed = load_records()
        drift = [i for i, r in built.items() if committed.get(i) != r]
        print(f"{'DRIFT ' + str(drift) if drift else 'records match the generator'}")
        return 1 if drift else 0
    print("usage: counterfactual.py --write | --check | --verify")
    return 2


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
