"""One source document cannot be two independent reasons (#1941).

The daily fact-improvement pass states its own thesis in the module docstring:
"every action here therefore needs an independent reason *on top of* the
detector's pairing". What it actually checked was *attributability*: the
confidence basis asks the higher side for a `source_doc` or a `created_at`
(`_has_attribution`) and never asks the two sides whether they came from
different places. Measured through `app.kg_store` on 2026-10-01, that predicate
discriminates nothing at all — `created_at` is on 121,112 of 121,112 active
`facts_idx` rows and `source_doc` on 121,103 of them — so every pair the pass can
plan against passes it.

The live witness the item names is entity `LLM Inference`: two rows both sourced
to `projects/lloyd/channel-eval/ai-engineer.md`, at confidence 0.95 against 0.85.
That gap is exactly `MIN_CONFIDENCE_GAP`, and it clears the floor only because
`_GAP_TOLERANCE` makes the floor `>=`. One document, two rows, and the plan reads
as a contest between two claims.

Nine of the ten nodes here are differentials; the tenth is the clause-6 witness
pin, and it is not one — see its own node for why that is the right shape. The
same-document fixtures (`LinFlagged`, `LinDry`, `LinApply`), the shared-hash one
(`LinHash`) and the differing-lineage one (`LinDiffering`) all come from one
`_pair` constructor, at the SAME confidences and the SAME ages, and differ in
nothing but their lineage fields.
The differential was measured, not asserted, and re-measured node by node on
this tree: with the veto's early-continue replaced by a no-op and the flag and
the counters left standing — the mutant that isolates this one branch — five
nodes go red (the three pinning a same-document pair, the shared-hash node, and
the record node) and five stay green (differing lineage, the sub-floor refusal,
write order, the refused entity, and the witness node). Against the module as it
stood before this veto landed, where none of the fields exist yet, nine go red
and only the witness node stays green, because that node reads the committed
vault bytes and never touches a field this change added — it is the one node
here a revert of the veto cannot break, which is the price of pinning a clause
about bytes held in another repo. That 5-red / 5-green split is the only
arrangement that shows a withheld action was withheld *by lineage* and not by
the gap floor, the god-node bound, or a fixture that never paired.

Two numbers are pinned beside the behaviour, because the record has to be able to
say what the veto cost: `lineage_pairs` (exposure) and `lineage_withheld` /
`lineage_withheld_pairs` (what was declined). A withheld count of 0 with no
exposure beside it is the #1348 failure mode again — "reports every entity and
plans nothing reads as a healthy night".

The veto is on the confidence basis only, which is why one node here asserts the
OPPOSITE outcome: a same-document pair at equal confidence and a day between the
writes is write-order evidence, and a later write inside one document is exactly
what that basis exists to act on.

Where the measured figures come from (clause 6). The witness is a run record in
the VAULT repo, not this one — `~/obsidian/backlog/data/20260930-210019-dryrun.json`,
landed by that repo's commit `7756b170` (resolve it with
`git -C ~/obsidian log -1 7756b170`; no such object exists in this repo, and the
hash is named as the vault's, not as a commit here). It is byte-identical to the
run it copies, `~/lloyd-data/_pipeline/improvement/20260930-210019-dryrun.json`,
`cmp` clean. `test_the_committed_witness_bytes_still_say_one_action_in_40` re-derives
the figures from those committed bytes: 855 newlines, `actions_planned: 1`, 40
entities. It carries no `live_vault` marker on purpose, following
`tests/test_archived_skill_artifacts.py:30-41` — a marked assertion about vault
content is deselected by the gate's own `-m "not live_vault"` and so enforces
nothing — and the coupling that buys is acceptable only because this witness is a
frozen historical run that no nightly job rewrites. If the file is missing the node
fails naming the path; it does not skip, because a guard that cannot see its input
has no verdict to report.

The same bytes are the corpus of the pre-ship measurement (item clause 3), whose
own figures are NOT in the record and so are not pinned by that node: re-running
the shipped emit path on 2026-10-01 over the same 40 entities gave `pairs_before`
301 of which `lineage_pairs` 290, and the veto withheld 0 — admitted actions 1
before it and 1 after. The record's own `pairs_before` is 285, measured on the
2026-09-30 corpus; the two figures differ because the corpus moved in a day, and
the node below pins the record's number, not the re-run's.
"""

from __future__ import annotations

import datetime
import json
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent_mcp._shared as shared                     # noqa: E402
from agent_mcp import facts as facts_mod               # noqa: E402
from agent_mcp import retrieval                        # noqa: E402
from agent_mcp import fact_improvement as fi           # noqa: E402
from app import kg_store                               # noqa: E402

_DOC_A = "projects/lloyd/channel-eval/ai-engineer.md"
_DOC_B = "projects/lloyd/channel-eval/prefetch-cost.md"
_HASH_A = "a" * 64
_HASH_B = "b" * 64

# The pair, in the shape the detector really pairs on: the daemon-state
# opposition from the `Idcol` fixture in
# tests/test_fact_improvement_confidence_trace.py, whose `confidence` action is
# proven green on this tree. An invented pair does not pair — that file records
# `_detect_contradictions_sync` returning `contradictions: 0` for two
# same-subject numeric texts, and a fixture like that plans nothing and proves
# nothing.
#
# #2366 moved the two words. The pair is the same shape and still pairs, but the
# state has to be a word `_asserted_predicate_values` can read —
# `_PREDICATE_VALUE_RE` is `true|false|enabled|disabled`, and `active`/`inactive`
# are in `_OPPOSING_PAIRS` yet not in it — so a row that said only "is inactive"
# NAMES `lld.indexerd` without valuing it. Since #2366 a co-named token valued by
# neither side is no longer a supersession basis, which would have left this
# file's clause-4 node (the write-order basis is untouched) planning nothing and
# asserting a differential that was never running.
_LOSER_TEXT, _WINNER_TEXT = ("The daemon lld.indexerd is disabled.", "The daemon lld.indexerd is enabled.")
_GAP_LOSER_CONF, _GAP_WINNER_CONF = 0.3, 0.9
# Ages that would be a write-order story if the confidences were equal: 37 days
# apart, four times over MIN_STALE_GAP_DAYS. They are not the basis for these
# pairs, because the confidences differ, and the nodes below that need write
# order set the confidences equal instead.
_LOSER_AGE_DAYS, _WINNER_AGE_DAYS = 3, 40


def _iso(days_ago: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=days_ago)).isoformat()


def _fresh_tree(tmp_path, monkeypatch):
    """A facts root holding nothing but what the test writes, plus an empty store."""
    facts_root = tmp_path / "facts"
    (facts_root / "Unrelated").mkdir(parents=True)
    (facts_root / "Unrelated" / "Unrelated-state.md").write_text(
        "---\ntype: facts\nentity: Unrelated\ncategory: state\nfacts: []\n---\n",
        encoding="utf-8")
    for mod in (shared, facts_mod, retrieval, fi):
        monkeypatch.setattr(mod, "FACTS_ROOT", facts_root)
    monkeypatch.setattr(fi, "RECORD_DIR", tmp_path / "records")
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    st = kg_store.configure(tmp_path / "kg.sqlite")
    return facts_root, st


def _pair(entity: str, *, loser_doc, loser_hash, winner_doc, winner_hash,
          loser_conf: float = _GAP_LOSER_CONF, winner_conf: float = _GAP_WINNER_CONF,
          category: str = "state") -> list[dict]:
    """The two rows, differing only in the fields the caller names.

    Ids are a per-file counter on this tree, so both rows carry `fact-00N` as
    the real writers would give them; nothing here selects on id.
    """
    return [
        {"file": f"{entity}/{entity}-{category}.md", "id": "fact-001",
         "fact": _WINNER_TEXT, "confidence": winner_conf, "age_days": _WINNER_AGE_DAYS,
         "source_doc": winner_doc, "source_hash": winner_hash},
        {"file": f"{entity}/{entity}-{category}.md", "id": "fact-002",
         "fact": _LOSER_TEXT, "confidence": loser_conf, "age_days": _LOSER_AGE_DAYS,
         "source_doc": loser_doc, "source_hash": loser_hash},
    ]


def _write(root: Path, records: list[dict]) -> Path:
    """Group `records` by their `file` and write one fact file each, ids verbatim."""
    by_file: dict[Path, list[dict]] = {}
    for rec in records:
        by_file.setdefault(root / rec["file"], []).append(rec)
    written = None
    for path, group in by_file.items():
        category = path.name.split("-", 1)[1][:-3]
        entity = path.parent.name
        prepared = []
        for rec in group:
            created = _iso(rec["age_days"])
            prepared.append({"fact": rec["fact"], "confidence": rec["confidence"],
                             "category": category, "id": rec["id"],
                             "created_at": created, "valid_at": created,
                             "expired_at": None, "invalid_at": None,
                             "provenance": "EXTRACTED",
                             # Written only when the fixture names one, because
                             # `_has_attribution` and `_shared_lineage` both read
                             # an ABSENT key differently from a `None` one, and
                             # the rows the live writers leave behind are the
                             # absent kind (#1348).
                             **({"source_doc": rec["source_doc"]}
                                if rec.get("source_doc") else {}),
                             **({"source_hash": rec["source_hash"]}
                                if rec.get("source_hash") else {})})
        fm = {"type": "facts", "entity": entity, "category": category, "facts": prepared}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n",
                        encoding="utf-8")
        written = path
    return written


def _tree(tmp_path, monkeypatch, records):
    """Write the fixture and index it, returning `(root, store)`."""
    root, st = _fresh_tree(tmp_path, monkeypatch)
    _write(root, records)
    st.facts_idx.reindex(root=root)
    return root, st


def _records_on_disk(path: Path) -> dict:
    """{fact text: record} for one fact file, read back from disk."""
    fm = yaml.safe_load(path.read_text().split("---")[1])
    return {f["fact"]: f for f in fm["facts"]}


def _flagged_pairs_seen(monkeypatch) -> list[dict]:
    """Capture the detector's own pair dicts, so the flag has a witness.

    `plan_entity` returns counts, not pairs, so `shared_lineage: true` is
    otherwise observable only inside the pass. The dicts the detector returns ARE
    the dicts the plan loop walks, so whatever the loop flags is visible here
    afterwards — and this is the only way to tell "the pair was flagged and then
    declined" from "the pair was never looked at".
    """
    seen: list[dict] = []
    real = fi._detect_contradictions_sync

    def spy(entity, category=None, **kwargs):
        out = real(entity, category, **kwargs)
        seen.extend(out.get("contradictions", []))
        return out

    monkeypatch.setattr(fi, "_detect_contradictions_sync", spy)
    return seen


# ── clause 1: same `source_doc` → reported, never acted on ───────────────────

def test_same_source_doc_pair_is_flagged_and_counts_as_one_pair(tmp_path, monkeypatch):
    """The reported half of clause 1: the pair is seen, named, and still counted.

    "Never acted on" is worthless if the pair silently vanished, so this pins the
    other half first: one pair in, one pair in the returned contradiction count,
    and the flag on the detector's own pair dict.
    """
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinFlagged", loser_doc=_DOC_A, loser_hash=_HASH_A,
        winner_doc=_DOC_A, winner_hash=_HASH_A))
    seen = _flagged_pairs_seen(monkeypatch)
    plan = fi.plan_entity("LinFlagged")

    assert plan["pairs_before"] == 1, plan["pairs_before"]
    assert plan["contradictions"] == 1, plan["contradictions"]
    assert len(seen) == 1, seen
    assert seen[0].get("shared_lineage") is True, (
        "the veto has to say WHICH pairs it declined on the pair itself, or a "
        f"reader of the detector's list cannot tell it from a gap-floor decline: {seen[0]}")
    assert plan["lineage_pairs"] == 1, plan
    assert plan["lineage_withheld"] == 1, plan
    assert plan["actions"] == [], plan["actions"]
    # Nothing was written either, on disk and not by the plan's say-so. The
    # denominator is pinned first: a loop over records that failed to land
    # asserts nothing, which is the zero-denominator shape, so an unreadable file
    # would read here as "nothing was marked".
    on_disk = _records_on_disk(root / "LinFlagged" / "LinFlagged-state.md")
    assert len(on_disk) == 2, (
        f"both rows have to be on disk to show neither was marked: {on_disk}")
    for text, rec in on_disk.items():
        assert rec.get("expired_at") is None and rec.get("invalid_at") is None, (
            f"{text!r} was marked by a pass that owed it no action: {rec}")


def test_same_source_doc_pair_yields_no_confidence_action_in_dry_run(tmp_path, monkeypatch):
    """Clause 1, dry-run half: 0.9 against 0.3 on one document plans nothing."""
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinDry", loser_doc=_DOC_A, loser_hash=_HASH_A,
        winner_doc=_DOC_A, winner_hash=_HASH_A))
    rec = fi.run_improvement(entities=["LinDry"], record=False)

    gap = _GAP_WINNER_CONF - _GAP_LOSER_CONF
    assert gap >= fi.MIN_CONFIDENCE_GAP, (
        "the fixture must clear the floor the clause names, or it declines for "
        f"the wrong reason: gap {gap} vs floor {fi.MIN_CONFIDENCE_GAP}")
    assert rec["actions_planned"] == 0, rec["per_entity"]
    assert rec["per_entity"][0]["planned"] == 0, rec["per_entity"]
    assert rec["per_entity"][0]["contradictions"] == 1, rec["per_entity"]
    assert rec["lineage_withheld_pairs"] == 1, rec["lineage_withheld_pairs"]
    assert rec["actions_taken"] == 0, rec["actions_taken"]


def test_same_source_doc_pair_yields_no_confidence_action_under_apply(tmp_path, monkeypatch):
    """Clause 1, `apply=True` half: the withheld pair is untouched on disk.

    A dry-run-only veto would be a report, not a guard. The writes are what makes
    this clause need a second node: `apply_action` stamps `expired_at` into the
    fact markdown, so the only evidence that it did not is the markdown.
    """
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinApply", loser_doc=_DOC_A, loser_hash=_HASH_A,
        winner_doc=_DOC_A, winner_hash=_HASH_A))
    rec = fi.run_improvement(apply=True, entities=["LinApply"], record=False)

    assert rec["apply"] is True, rec["apply"]
    assert rec["writes_refused_reason"] is None, (
        "a run that refused every write because the store would not answer would "
        f"leave both rows untouched for a reason that is not this veto: "
        f"{rec['writes_refused_reason']}")
    assert rec["actions_planned"] == 0, rec["per_entity"]
    assert rec["actions_taken"] == 0, rec["actions_taken"]
    assert rec["lineage_withheld_pairs"] == 1, rec["lineage_withheld_pairs"]
    path = root / "LinApply" / "LinApply-state.md"
    on_disk = _records_on_disk(path)
    assert len(on_disk) == 2, (
        f"both rows have to be on disk to show neither was marked: {on_disk}")
    for text, marked in on_disk.items():
        assert marked.get("expired_at") is None and marked.get("invalid_at") is None, (
            f"apply marked {text!r} that the veto owes an action to: {marked}")
        assert "conflicts_with" not in marked, (
            f"apply wrote a contradiction trace onto {text!r}: {marked}")


# ── clause 2: same `source_hash`, differing `source_doc` ─────────────────────

def test_same_source_hash_with_differing_source_doc_is_withheld(tmp_path, monkeypatch):
    """Clause 2: one document cited by two paths is still one document.

    The `source_doc` strings differ, so a `source_doc`-only veto would act on
    this pair; the rows carry the same content hash, which is what makes them one
    origin. Checked through the same plan path as clause 1 — `plan_entity`, with
    the pair coming off the detector and not out of this file.
    """
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinHash", loser_doc=_DOC_B, loser_hash=_HASH_A,
        winner_doc=_DOC_A, winner_hash=_HASH_A))
    plan = fi.plan_entity("LinHash")

    assert plan["actions"] == [], plan["actions"]
    assert plan["pairs_before"] == 1, plan["pairs_before"]
    assert plan["lineage_pairs"] == 1, plan
    assert plan["lineage_withheld"] == 1, plan

    # And it is the HASH half that fired, not the doc half, on rows read back
    # through the real view rather than built here.
    view = retrieval.get_facts_sync("LinHash", with_source_file=True).get("facts", [])
    assert len(view) == 2, view
    field = fi._shared_lineage(view[0], view[1])
    assert field == "source_hash", (
        f"differing source_doc rows must be joined by the hash or not at all: {field}")


# ── clause 3: differing lineage still acts, unchanged ────────────────────────

def test_differing_lineage_pair_still_yields_its_confidence_action(tmp_path, monkeypatch):
    """Clause 3: same facts, different documents, action intact.

    The differential this file is built on. `_pair` is the same constructor the
    withheld nodes call, at the same confidences and ages, with both lineage
    fields differing — so the only thing that changed between this node and
    `test_same_source_doc_pair_yields_no_confidence_action_in_dry_run` is whether
    the two rows trace to one document. That is what makes the veto a lineage
    condition rather than a second, stricter gap floor.
    """
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinDiffering", loser_doc=_DOC_B, loser_hash=_HASH_B,
        winner_doc=_DOC_A, winner_hash=_HASH_A))
    plan = fi.plan_entity("LinDiffering")

    assert plan["lineage_pairs"] == 0, plan
    assert plan["lineage_withheld"] == 0, plan
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "confidence", action
    assert action["loser_fact"] == _LOSER_TEXT, action
    assert action["winner_fact"] == _WINNER_TEXT, action
    # The reason string is the one the pass wrote before this change: the veto
    # removed nothing from it, and `shared_lineage` is not a reason.
    assert f"{_GAP_LOSER_CONF} < {_GAP_WINNER_CONF}" in action["reason"], action["reason"]
    assert "lineage" not in action["reason"], action["reason"]


def test_a_sub_floor_gap_is_the_floor_s_refusal_and_not_charged_to_the_veto(
        tmp_path, monkeypatch):
    """Clause 3's other half, and clause 5's attribution: 0.95 vs 0.90 costs nothing here.

    Differing lineage, gap 0.05 — below `MIN_CONFIDENCE_GAP`, so the pair was
    always going to be declined, by the floor. If the withheld counter swept up
    every declined pair it would report a veto that never fired, and the record
    would blame this change for an action rate the floor was already producing.
    """
    root, st = _tree(tmp_path, monkeypatch, _pair(
        "LinSubFloor", loser_doc=_DOC_B, loser_hash=_HASH_B,
        winner_doc=_DOC_A, winner_hash=_HASH_A,
        loser_conf=0.9, winner_conf=0.95))
    rec = fi.run_improvement(entities=["LinSubFloor"], record=False)

    assert 0.95 - 0.9 < fi.MIN_CONFIDENCE_GAP, "the fixture must sit under the floor"
    assert rec["actions_planned"] == 0, rec["per_entity"]
    assert rec["per_entity"][0]["contradictions"] == 1, rec["per_entity"]
    assert rec["lineage_pairs"] == 0, rec["lineage_pairs"]
    assert rec["lineage_withheld_pairs"] == 0, (
        "a pair declined by the gap floor must not be charged to the lineage "
        f"veto, or the field stops being this change's cost: {rec['lineage_withheld_pairs']}")


# ── clause 4: the write-order basis is untouched ─────────────────────────────

def test_a_same_document_pair_still_supersedes_on_write_order(tmp_path, monkeypatch):
    """Clause 4: one document CAN correct itself, and that is the other basis's job.

    Equal confidences and 37 days between the writes, both rows on
    `_DOC_A`, and both rows naming `lld.indexerd` — the predicate both facts
    co-name, which since #2078 is what the age basis stands on. Since #2366 the
    two rows also have to VALUE it (`disabled` then `enabled`): a token neither
    text values names no assertion to correct and admits nothing, so this node
    would otherwise be asserting a differential it never ran.
    `MIN_CONFIDENCE_GAP` cannot fire on equal confidences, so the basis here is
    `_loser_by_age` plus that co-named predicate, and a later write inside one
    document is the whole evidentiary value of write order. A pair-level veto
    applied to both bases would block exactly this action, which is why the
    lineage check sits inside the `c1 != c2` branch and not above it.
    """
    records = _pair("LinSupersede", loser_doc=_DOC_A, loser_hash=_HASH_A,
                    winner_doc=_DOC_A, winner_hash=_HASH_A,
                    loser_conf=0.9, winner_conf=0.9)
    root, st = _tree(tmp_path, monkeypatch, records)
    plan = fi.plan_entity("LinSupersede")

    assert plan["lineage_pairs"] == 1, (
        f"the pair shares a document and must say so: {plan}")
    assert plan["lineage_withheld"] == 0, (
        f"the veto owes the write-order basis nothing: {plan}")
    assert len(plan["actions"]) == 1, plan["actions"]
    action = plan["actions"][0]
    assert action["kind"] == "superseded", action
    # The OLDER row is the superseded one, and here the older row is the
    # `active` text — the row the confidence nodes above call the winner. Which
    # side loses is the planner's choice off `created_at`, not this file's.
    assert action["loser_fact"] == _WINNER_TEXT, action

    rec = fi.run_improvement(apply=True, entities=["LinSupersede"], record=False)
    assert rec["actions_taken"] == 1, rec["per_entity"]
    assert rec["lineage_withheld_pairs"] == 0, rec["lineage_withheld_pairs"]
    marked = _records_on_disk(root / "LinSupersede" / "LinSupersede-state.md")
    assert marked[_WINNER_TEXT].get("expired_at"), (
        f"the veto blocked a legitimate self-supersede: {marked[_WINNER_TEXT]}")
    assert marked[_LOSER_TEXT].get("expired_at") is None, marked[_LOSER_TEXT]


# ── clause 5: the record states the cost ─────────────────────────────────────

def test_the_run_record_separates_withheld_from_clean(tmp_path, monkeypatch):
    """Clause 5: one run, one withheld entity and one clean one, read off the record.

    Both numbers, in one record, from two entities that differ only in lineage:
    non-zero where a pair was declined, 0 where none was. The clause asks that an
    action-rate drop be attributable from the record alone, which needs the zero
    to be as real as the one.
    """
    root = tmp_path / "facts"
    root.mkdir(parents=True)
    (root / "Unrelated").mkdir()
    (root / "Unrelated" / "Unrelated-state.md").write_text(
        "---\ntype: facts\nentity: Unrelated\ncategory: state\nfacts: []\n---\n",
        encoding="utf-8")
    for mod in (shared, facts_mod, retrieval, fi):
        monkeypatch.setattr(mod, "FACTS_ROOT", root)
    monkeypatch.setattr(fi, "RECORD_DIR", tmp_path / "records")
    shared._invalidate_entity_dirs_cache()
    retrieval.invalidate_fact_file_cache()
    retrieval._entity_index_cache = None
    st = kg_store.configure(tmp_path / "kg.sqlite")
    _write(root, _pair("LinRecVetoed", loser_doc=_DOC_A, loser_hash=_HASH_A,
                       winner_doc=_DOC_A, winner_hash=_HASH_A))
    _write(root, _pair("LinRecClean", loser_doc=_DOC_B, loser_hash=_HASH_B,
                       winner_doc=_DOC_A, winner_hash=_HASH_A))
    st.facts_idx.reindex(root=root)

    rec = fi.run_improvement(entities=["LinRecVetoed", "LinRecClean"], record=True)

    by_entity = {e["entity"]: e for e in rec["per_entity"]}
    assert by_entity["LinRecVetoed"]["lineage_withheld"] == 1, by_entity
    assert by_entity["LinRecClean"]["lineage_withheld"] == 0, by_entity
    assert by_entity["LinRecVetoed"]["lineage_pairs"] == 1, by_entity
    assert by_entity["LinRecClean"]["lineage_pairs"] == 0, by_entity
    assert rec["lineage_withheld_pairs"] == 1, rec["lineage_withheld_pairs"]
    assert rec["lineage_pairs"] == 1, rec["lineage_pairs"]
    # The action rate the withheld pair cost, readable beside it.
    assert rec["actions_planned"] == 1, rec["actions_planned"]

    # And the same two numbers survive the JSON round-trip into the written
    # record, which is the artifact the owed-check job reads on later nights.
    written = json.loads(Path(rec["record_path"]).read_text(encoding="utf-8"))
    assert written["lineage_withheld_pairs"] == 1, written["lineage_withheld_pairs"]
    assert written["lineage_pairs"] == 1, written["lineage_pairs"]
    assert {e["entity"]: e["lineage_withheld"] for e in written["per_entity"]} == {
        "LinRecVetoed": 1, "LinRecClean": 0}, written["per_entity"]


def test_a_refused_entity_reports_unknown_lineage_not_zero(tmp_path, monkeypatch):
    """The zero has to mean zero, so an unscanned entity may not borrow it (#702).

    An over-bound entity was never pairwise scanned, so its lineage is unknown.
    A 0 there would read as "scanned, nothing shared an origin", which is the
    same misreading that made `contradictions: 0, refused: true` and a clean
    entity byte-identical.
    """
    root, st = _tree(tmp_path, monkeypatch, [
        {"file": "LinBound/LinBound-state.md", "id": f"fact-{i:03d}",
         # n alternating between two opposing texts: the detector pairs every
         # (active, inactive) combination, so a fixture over the bound is also a
         # fixture holding nothing but shared-lineage pairs.
         "fact": ("The daemon is active." if i % 2 else "The daemon is inactive."),
         "confidence": 0.9, "age_days": 5,
         "source_doc": _DOC_A, "source_hash": _HASH_A}
        for i in range(1, facts_mod.FACT_GODNODE_THRESHOLD + 2)])
    rec = fi.run_improvement(entities=["LinBound"], record=False)
    entry = rec["per_entity"][0]

    assert entry["refused"] is True, entry
    assert entry["lineage_pairs"] is None, (
        f"an unscanned entity must not report a zero lineage count: {entry}")
    assert entry["lineage_withheld"] is None, entry
    assert rec["lineage_withheld_pairs"] == 0, rec["lineage_withheld_pairs"]



# ── clause 6: the witness bytes the item quotes ──────────────────────────────

#: The run record `#1941` cites, committed to the VAULT repo (its own git
#: history, commit `7756b170` there — resolvable with `git -C ~/obsidian log -1
#: 7756b170`, and deliberately not as a commit in this repo, where no such
#: object exists). A frozen historical run, so unlike a skills index or a daily
#: note it cannot drift under a nightly rewrite.
WITNESS = Path.home() / "obsidian" / "backlog" / "data" / "20260930-210019-dryrun.json"


def test_the_committed_witness_bytes_still_say_one_action_in_40():
    """Clause 6: the figures the item quotes are re-derived, not remembered.

    The item and its triage both quote this record — `actions_planned: 1` over 40
    entities — to show the confidence basis admitted nothing on the current
    corpus, which is what makes clause 3's "if the veto removes most actions, do
    not ship it" checkable. Until now those bytes lived only outside git
    (`~/lloyd-data/_pipeline/improvement/…`, a directory with no history), so the
    sentence rested on a file that could change with no record. This node reads
    the committed copy and pins the four facts the claim rests on, including the
    one that makes a withheld count of 0 meaningful there: the single planned
    action is on the `superseded` basis, the basis this veto does not touch, so 0
    withheld means the confidence basis admitted nothing, not that the veto
    silently ate work.

    Unmarked, not `live_vault`, per `tests/test_archived_skill_artifacts.py:30-41`:
    a marked vault assertion is deselected by the gate's own `-m "not live_vault"`
    and so can never enforce anything. The coupling that buys — this node is red
    on any machine without that vault file — is paid for by failing loudly with
    the path named, never by skipping.
    """
    assert WITNESS.exists(), (
        f"clause 6's witness is not on disk at {WITNESS}. A missing witness is "
        "not a passing one: the item's figure would rest on nothing. It is "
        "committed to the vault repo as backlog/data/20260930-210019-dryrun.json.")
    raw = WITNESS.read_text(encoding="utf-8")
    # `wc -l` semantics, since 855 is the figure `wc -l <
    # backlog/data/20260930-210019-dryrun.json` yields and the one the clause
    # names. `splitlines()` is 856 on these bytes, so the count is of newlines,
    # not of lines-as-severed.
    assert raw.count("\n") == 855, (
        f"the committed witness is {raw.count(chr(10))} newlines, not the 855 "
        "the clause quotes — the copy has been replaced, not preserved")

    rec = json.loads(raw)
    assert rec["apply"] is False, (
        "the witness is a dry-run record; an applied run would not carry the "
        f"planned-but-not-taken shape the item quotes: apply={rec['apply']}")
    assert rec["actions_planned"] == 1, rec["actions_planned"]
    assert len(rec["entities"]) == 40, len(rec["entities"])
    # Not a stub: the record's own pair count over those entities, which a
    # hand-written stand-in would not reproduce.
    assert rec["pairs_before"] == 285, rec["pairs_before"]

    planned = [(e["entity"], a["kind"])
               for e in rec["per_entity"] for a in e.get("actions", [])]
    assert planned == [("TTS", "superseded")], planned
    assert rec["actions_taken"] == 0, rec["actions_taken"]
