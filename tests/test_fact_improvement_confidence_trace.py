"""The daily fact-improvement loop's contradiction loser has to name its winner.

Two writers stamp contradiction-loser semantics onto a fact record's `invalid_at`.
`agent_mcp/facts.py::_fact_resolve_apply` — the MCP tool, which has no automated
caller (#463 closed that question by deletion) — marks by (file, id) and passes
`extra_fields`, so it leaves a `conflicts_with` record naming the winner. The daily
fact-improvement loop (autonomy task #84) marked the SAME field for its `confidence`
actions and passed nothing, because it reached the fact through a text match and
`_apply_fact_marks` attaches extras only on the identity path — a documented
restriction (`facts.py:325-331`, enforced at `:394`), not a bug to widen here.

So the nightly line `Fact records carrying a 'conflicts_with' resolution trace |
0 of 119,167 fact records` counted the writer that never runs and was structurally
blind to the one that does. The fix carries the winner the plan step already
computed into the action, aims a confidence mark by (file, id), and puts the trace
in the same atomic write as the mark. Every node below reads the fact record back off
disk — the returned dict is never the evidence.

Before this change every node here fails: the mark lands with no trace, and the
result carries no `traces_written` at all.
"""

from __future__ import annotations

import datetime
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


def _iso(days_ago: int) -> str:
    return (datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=days_ago)).isoformat()


# The pair, in the shape the detector really pairs on — the `Idcol` set from
# tests/test_fact_identity_one_action_one_fact.py, whose `confidence` action is proven
# green on this tree. A pair of my own invention does not pair: measured directly,
# `_detect_contradictions_sync` returned `contradictions: 0` for two same-subject
# numeric texts, so a fixture like that plans nothing and proves nothing.
#
# The two files deliberately REUSE BOTH IDS. Ids are a per-file counter
# (`app.fact_ids.next_fact_id`), so `fact-001` here names two different facts: that is
# the collision that made one call invalidate 25 facts to change 2 (#874), and the
# reason the mark's key is (file, id) rather than id. The trace has to name a file for
# the same reason — a bare `fact-001` in a trace would be true of two facts.
#
# Which of these the planner condemns is not decided here. The nodes below take the
# action `plan_entity` returns and read the winner's address off the attributed view,
# so the fixture cannot pass by asserting a layout the planner did not choose.
# Measured against the planner, not guessed: `plan_entity("Idcol")` condemns the
# 0.3-confidence `usage/fact-002` and keeps its 0.9 twin in the SAME file, which also
# means `state/fact-001` and `state/fact-002` — the other pair this fixture holds —
# must come out untouched.
_LOSER_FILE = "Idcol/Idcol-usage.md"
_WINNER_FILE = "Idcol/Idcol-usage.md"
_TWIN_FILE = "Idcol/Idcol-state.md"        # the id twins that must stay untouched
_LOSER_TEXT, _WINNER_TEXT = ("The daemon is inactive.", "The daemon is active.")
_LOSER_ID, _WINNER_ID = "fact-002", "fact-001"

_RECORDS = [
    {"file": "Idcol/Idcol-state.md", "id": "fact-001",
     "fact": "The indexer is disabled.", "confidence": 0.3, "age_days": 40},
    {"file": "Idcol/Idcol-state.md", "id": "fact-002",
     "fact": "The indexer is enabled.", "confidence": 0.9, "age_days": 3},
    {"file": "Idcol/Idcol-usage.md", "id": "fact-001",
     "fact": "The daemon is active.", "confidence": 0.9, "age_days": 40},
    {"file": "Idcol/Idcol-usage.md", "id": "fact-002",
     "fact": "The daemon is inactive.", "confidence": 0.3, "age_days": 3},
]


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


def _write(root: Path, records: list[dict], entity: str = "Idcol") -> list[Path]:
    """Group `records` by their `file` and write one fact file each, ids verbatim."""
    by_file: dict[Path, list[dict]] = {}
    for rec in records:
        by_file.setdefault(root / rec["file"], []).append(rec)
    written = []
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
                             "provenance": "STATED", "source_doc": None})
        fm = {"type": "facts", "entity": entity, "category": category,
              "facts": prepared}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n",
                        encoding="utf-8")
        written.append(path)
    return written


def _records(path: Path) -> dict:
    """{id: record} for one fact file, read back from disk."""
    fm = yaml.safe_load(path.read_text().split("---")[1])
    return {f["id"]: f for f in fm["facts"]}


def _built_tree(tmp_path, monkeypatch, records):
    root, st = _fresh_tree(tmp_path, monkeypatch)
    _write(root, records)
    st.facts_idx.reindex(root=root)
    return root


def _planned_confidence_action(tmp_path, monkeypatch):
    """The action the real planner produces, with nothing restated by this file.

    Returns `(root, action)`. Building the action by hand would let this file pass on
    fields the planner does not emit, which is exactly the half clause 2 is about.
    """
    root = _built_tree(tmp_path, monkeypatch, _RECORDS)
    planned = fi.plan_entity("Idcol")
    action = next((a for a in planned["actions"] if a["kind"] == "confidence"), None)
    assert action is not None, (
        "the fixture must produce a confidence action; a plan that yields none "
        f"proves nothing about any clause: {planned['actions']}")
    return root, action


def _confidence_action() -> dict:
    """A confidence action aimed at the condemned record by (file, id).

    `category` is read off the loser's own file name rather than written out, because
    `_aim_substring` counts its probe inside that category — a hand-written category
    that disagreed with the file would make the action skip, and a skipped action
    proves nothing about a trace.
    """
    return {"kind": "confidence", "entity": "Idcol",
            "category": Path(_LOSER_FILE).name.split("-", 1)[1][:-3],
            "loser_fact": _LOSER_TEXT, "loser_id": _LOSER_ID,
            "loser_source_file": _LOSER_FILE,
            "winner_fact": _WINNER_TEXT, "winner_id": _WINNER_ID,
            "winner_source_file": _WINNER_FILE,
            "reason": "newer claim contradicts older at lower confidence"}


def _superseded_action() -> dict:
    return dict(_confidence_action(), kind="superseded",
                reason="newer claim supersedes older at write time")


def _apply(tmp_path, monkeypatch, action, *, extra_records=()):
    """Write the fixture, apply `action`, return `(root, loser_path, result)`."""
    root = _built_tree(tmp_path, monkeypatch, list(_RECORDS) + list(extra_records))
    result = fi.apply_action(action, _iso(1))
    return root, root / Path(action["loser_source_file"]), result


def _spy_on_mark_writer(monkeypatch):
    """Record the arguments of every `_apply_fact_marks` call, and pass them through.

    The disk says what a mark LOOKS like; only the call says how it was aimed. Two
    clauses turn on that distinction — whether an extra field travelled with the mark,
    and whether the mark was aimed by (file, id) or by a phrase — and neither is
    recoverable afterwards from the fact record.
    """
    calls: list[dict] = []
    real = fi._apply_fact_marks

    def spy(marks, files, **kwargs):
        calls.append({"marks": dict(marks),
                      "extra_fields": kwargs.get("extra_fields"),
                      "identity_aim": kwargs.get("identity_ids") is not None})
        return real(marks, files, **kwargs)

    monkeypatch.setattr(fi, "_apply_fact_marks", spy)
    return calls


def _load_mark_writer(monkeypatch):
    """Spy `facts.atomic_write_text` — the one statement that puts a marked record down.

    `facts.py:409` is the only such statement inside `_apply_fact_marks`, and it sits
    within `locked_file(fact_file)`. Counting its calls is how "in one write" is graded
    without reading the writer's source — and a patch that named a function nobody calls
    would report a vacuous 1, which is why the nodes below pair it with a control
    (a rerun that changes nothing must add no call).
    """
    calls: list[str] = []
    real = facts_mod.atomic_write_text

    def spy(path, text, **kw):
        calls.append(str(path))
        return real(path, text, **kw)

    monkeypatch.setattr(facts_mod, "atomic_write_text", spy)
    return calls


# ── clause 1: the trace lands with the mark, in one write ────────────────────

def test_a_planned_confidence_apply_lands_the_trace_with_the_mark_in_one_write(
        tmp_path, monkeypatch):
    """The real planner's action, applied: loser marked, winner named, one write."""
    root, action = _planned_confidence_action(tmp_path, monkeypatch)
    assert action["winner_id"] != action["loser_id"], (
        "the fixture's winner and loser became one record, so a trace pointing the "
        "loser at itself would still read as well-formed")
    loser_file = root / Path(action["loser_source_file"])
    calls = _load_mark_writer(monkeypatch)

    result = fi.apply_action(action, _iso(1))
    assert result["expired_count"] == 1, result
    assert result["traces_written"] == 1, result
    assert calls.count(str(loser_file)) == 1, (
        f"mark and trace were not one write: {len(calls)} write(s) to "
        f"{loser_file.name} — {calls}")

    rec = _records(loser_file)[action["loser_id"]]
    assert rec["invalid_at"], result
    assert any(action["reason"] in line for line in rec["invalid_reason"]), rec
    assert rec["conflicts_with"]["type"] == "conflicts_with"
    assert rec["conflicts_with"]["file"] == action["winner_source_file"]
    assert rec["conflicts_with"]["fact_id"] == action["winner_id"]
    assert rec["conflicts_with"]["fact"] == action["winner_fact"]
    assert not Path(rec["conflicts_with"]["file"]).is_absolute(), (
        "an absolute path in the trace would break a rebuild pointed at another "
        "root — `retrieval.fact_source_file` spells it relative for that reason")
    # The whole file was re-serialised, so a sibling field lost in the rewrite would
    # read as a clean trace over a silently shortened record.
    assert rec["provenance"] == "STATED" and rec["valid_at"]
    # One fact, not two. `state/fact-002` shares the LOSER's id and is a claim about a
    # different thing entirely, so a mark aimed by id alone would take it out — that is
    # the #874 shape, one call invalidating 25 facts to change 2.
    twin = _records(root / _TWIN_FILE)[_LOSER_ID]
    assert twin["fact"] != rec["fact"], (
        "the fixture's id twin became the same claim, so an id-only aim would look "
        "correct here and be wrong on the live store")
    assert not twin.get("invalid_at") and "conflicts_with" not in twin, (
        f"the id twin in the other category file was marked: {twin}")
    winner = _records(root / Path(action["winner_source_file"]))[action["winner_id"]]
    assert not winner.get("invalid_at") and not winner.get("conflicts_with"), (
        f"the winner was marked too: {winner}")


def test_the_trace_and_the_mark_travel_in_the_same_call(tmp_path, monkeypatch):
    """`extra_fields` reaches `_apply_fact_marks` WITH the mark, not after it.

    Asserted at the seam where the two halves of clause 1 meet. A caller that passed
    extras on a text route could satisfy every on-disk assertion above only by
    accident, because the fixture's decoy would have absorbed the mark.
    """
    seen: dict = {}
    real = fi._apply_fact_marks

    def spy(marks, files_to_scan, **kw):
        seen.update({"marks": dict(marks), "extras": kw.get("extra_fields"),
                     "text_matches": kw.get("text_matches")})
        return real(marks, files_to_scan, **kw)

    monkeypatch.setattr(fi, "_apply_fact_marks", spy)
    _root, loser_file, result = _apply(tmp_path, monkeypatch, _confidence_action())
    assert result["expired_count"] == 1, result
    assert len(seen["marks"]) == 1, seen["marks"]
    key = next(iter(seen["marks"]))
    assert key == (_LOSER_FILE, _LOSER_ID), f"not aimed by identity: {key}"
    assert seen["extras"] is not None and set(seen["extras"]) == {key}, (
        "the extras did not travel with this mark")
    assert list(seen["extras"][key]) == ["conflicts_with"]
    assert seen["text_matches"] is None, (
        "an identity-aimed action still carried a text route, which would mark the "
        "decoy sharing the winner's sentence and leave that mark untraced")


def test_the_traced_loser_is_counted_by_the_health_report(tmp_path, monkeypatch):
    """Acceptance: `contradiction_trace_coverage` returns traced >= 1 after the apply.

    Crosses the boundary the nightly job actually crosses — a facts directory read by
    the report script — rather than calling a function inside the writer.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "khr_trace", ROOT / "scripts/memory/knowledge-health-report.py")
    khr = importlib.util.module_from_spec(spec)
    sys.modules["khr_trace"] = khr
    spec.loader.exec_module(khr)

    root = _built_tree(tmp_path, monkeypatch, _RECORDS)

    def traced():
        return khr.contradiction_trace_coverage(khr.load_entities(root))

    assert traced() == (0, 4), (
        f"the corpus was not trace-free before the apply: {traced()}")
    action = next(a for a in fi.plan_entity("Idcol")["actions"]
                  if a["kind"] == "confidence")
    assert fi.apply_action(action, _iso(1))["expired_count"] == 1

    after = traced()
    assert after[1] == 4, f"the population moved under a single-fact action: {after}"
    assert after[0] >= 1, f"the nightly line would still print 0 of 4: {after}"


# ── clause 2: the plan carries the winner as a key, not as prose ─────────────

def test_the_planned_confidence_action_carries_the_winner_file_and_id(
        tmp_path, monkeypatch):
    """Apply-time can name the winner without re-deriving it.

    Three keys, not one blob: the file and the id are what a mark's identity key and
    the trace's `file`/`fact_id` are built from; the text is the claim a reader reads
    in place of opening the winner's file. `reason` alone left apply-time with a
    sentence, which is the state that produced `0 of 119,167`.
    """
    _root, action = _planned_confidence_action(tmp_path, monkeypatch)
    assert action["kind"] == "confidence"
    for key in ("winner_source_file", "winner_id", "winner_fact"):
        assert key in action, f"{key} missing; apply-time could not name the winner"
    assert action["winner_source_file"] == _WINNER_FILE, action
    assert action["winner_id"] == _WINNER_ID, action
    assert action["winner_fact"] == _WINNER_TEXT, action
    assert action["loser_source_file"] == _LOSER_FILE
    # The keys are a NAME, not a copy of the loser's address: the ids collide across
    # these two files on purpose, so a trace that reused the loser's pair would point
    # the loser at itself and still look well-formed.
    assert (action["winner_source_file"], action["winner_id"]) \
        != (action["loser_source_file"], action["loser_id"])
    # Whole claim, not the 70-character excerpt the reason carries for a reader.
    assert action["winner_fact"] == action["winner_fact"].strip()


def test_an_action_from_an_unattributed_view_carries_no_winner_key(
        tmp_path, monkeypatch):
    """The empty keys are the plan's own emptiness, not a fixture's.

    `_plan_action` writes `winner.get("source_file") or ""` and `winner.get("id")` on
    whatever view it was handed, so a view that attributes a record to no file plans an
    action that cannot be traced — the case clause 3 grades at apply time.
    """
    root = _built_tree(tmp_path, monkeypatch, _RECORDS)
    real = fi._get_facts_sync
    monkeypatch.setattr(fi, "_get_facts_sync",
                        lambda entity, category=None, **kw: real(entity, category))
    planned = fi.plan_entity("Idcol")
    monkeypatch.undo()
    action = next((a for a in planned["actions"] if a["kind"] == "confidence"), None)
    assert action is not None, (
        "the plan must still reach a confidence action from an unattributed view, or "
        f"this node proves nothing: {planned['actions']}")
    assert action["loser_source_file"] == "" and action["winner_source_file"] == "", action
    # An ID survives the unattributed read; a FILE does not. That is exactly why a bare
    # id is not an address: #874 was one call invalidating 25 facts to change 2, ids
    # being a per-file counter, and `fact-001` in this fixture names a record in EACH
    # category file. Which is why `_confidence_trace` asks for both halves the way
    # `fact_identity` does, and why applying this action marks with no trace beside it.
    assert action["winner_id"], action
    assert root.exists()


# ── clause 3: no named winner, no trace; no mark, no trace ───────────────────

def test_a_confidence_action_with_an_unnamed_winner_marks_and_writes_no_trace(
        tmp_path, monkeypatch):
    """A loser with no attribution is still invalidated, and stays untraceable.

    Both halves of the rule in its own order: the mark lands (on the text route, which
    is the only one that can reach an unattributed fact), and no trace is written —
    plus the `untraceable` count, the only sign that a resolution happened and cannot
    be checked.
    """
    action = dict(_confidence_action(), loser_source_file="", winner_source_file="",
                  winner_id=None)
    root, _loser, result = _apply(tmp_path, monkeypatch, action)
    assert result["expired_count"] == 1, result
    assert result["traces_written"] == 0, result
    assert result["untraceable"] == 1, result

    rec = _records(root / _LOSER_FILE)[_LOSER_ID]
    assert rec["invalid_at"], result
    assert "conflicts_with" not in rec, (
        "a trace naming no file and no id would record a resolution nobody can "
        f"check, and the health report would count it as one: {rec}")


def test_a_confidence_action_that_marks_nothing_writes_no_trace(tmp_path, monkeypatch):
    """An action whose aim misses writes nothing and reports no trace.

    The aim names `fact-777`, which no record carries, and the text route is not taken
    on this path — so nothing is marked, and the two counts must stay at zero together.
    """
    action = dict(_confidence_action(), loser_id="fact-777")
    root, _loser, result = _apply(tmp_path, monkeypatch, action)
    assert result["expired_count"] == 0, result
    assert result["traces_written"] == 0, result
    assert result["untraceable"] == 0, (
        "nothing was marked, so nothing is untraceable; the counts must not move "
        "together")

    for rec in _records(root / _LOSER_FILE).values():
        assert not rec.get("invalid_at"), "an action that marked nothing stamped a fact"
        assert "conflicts_with" not in rec


def test_a_second_apply_over_the_same_pair_writes_no_second_trace(
        tmp_path, monkeypatch):
    """A rerun over one pair adds no trace to the store and reports zero written.

    Measured on disk first, which is the property the nightly figure depends on: after
    two applies over the same pair the file still holds exactly ONE traced record, and
    the mark writer was never called the second time.

    Which guard stops the rerun is asserted too, and on this tree it is the EARLIER
    one: the loser is no longer an active fact, so `_aim_substring` finds no unique
    match and the action returns before the mark. `_apply_fact_marks`'
    `already_marked` path — the one `_fact_resolve_apply:904-907` counts traces over —
    is therefore not reached here, and this node does not claim it is. The finding is
    on #1817 rather than hidden in this docstring.
    """
    root, loser_file, first = _apply(tmp_path, monkeypatch, _confidence_action())
    assert first["traces_written"] == 1, first
    calls = _load_mark_writer(monkeypatch)

    second = fi.apply_action(_confidence_action(), _iso(1))
    assert second["expired_count"] == 0, second
    assert "skipped" in second, (
        f"the rerun took a path this node does not describe: {second}")
    assert second["traces_written"] == 0, (
        "one pair was counted twice — which is how `0 of N` becomes `2 of N` with no "
        "second resolution having happened")
    assert second["untraceable"] == 0, second
    assert calls == [], f"the rerun wrote the file anyway: {calls}"
    traced = [r for path in sorted(root.rglob("*.md"))
              for r in _records(path).values() if r.get("conflicts_with")]
    assert len(traced) == 1, f"a rerun left {len(traced)} traces: {traced}"


# ── clause 4: write order is a different judgment ────────────────────────────

def test_a_planned_superseded_action_keeps_expired_at_and_gains_no_trace(
        tmp_path, monkeypatch):
    """Clause 4: write order is a different judgment, and the fix leaves it alone.

    Produced by `plan_entity`, not hand-built: `kind == "superseded"` is a branch of
    the planner (`_age_basis`), and the action this node applies is the one that branch
    returns. Getting there needs a pair the detector pairs on text that the confidence
    basis then refuses: these two are opposing-terms at EQUAL confidence — the only
    ground the age basis stands on, since `_plan_action` skips a pair whose confidences
    differ but do not clear `MIN_CONFIDENCE_GAP` (0.1) — and 37 days apart, above
    `MIN_STALE_GAP_DAYS`. The older claim is the superseded one, and it is the loser
    here even though its confidence is the same as the winner's. A pair that produced a
    `confidence` action instead would let this node pass on the wrong branch.

    What must not change: a superseded mark is `expired_at`/`expire_reason` ("was true,
    no longer is"), never `invalid_at` ("should not have been recorded"), and it gains
    no `conflicts_with`. #1817 extends the confidence route only; widening it here would
    record a write-order cleanup as a contradiction the store cannot defend.
    """
    root, action = _planned_superseded_action(tmp_path, monkeypatch)
    assert action["kind"] == "superseded", action

    mark_calls = _spy_on_mark_writer(monkeypatch)
    result = fi.apply_action(action, _iso(1))
    assert result["expired_count"] == 1, result
    assert result["marked"] and result["expired_count"] == len(result["marked"]), result
    assert result["field"] == "expired_at", result
    assert result["traces_written"] == 0 and result["traces"] == [], result
    # And it is not a trace that was built and then dropped from the count: the call
    # that did the marking carried no extras and no identity aim at all, which is what
    # makes it impossible for this branch to grow a `conflicts_with` later by accident.
    assert mark_calls == [
        {"marks": {}, "extra_fields": None, "identity_aim": False}], (
        f"a superseded mark reached the writer with trace machinery: {mark_calls}")
    # `untraceable` is a contradiction-accounting field. A write-order action that
    # marked one fact is not an untraceable resolution, and counting it as one would
    # put cleanups into the figure the nightly line reads as uncheckable.
    assert result["untraceable"] == 0, result

    loser_file = root / Path(action["loser_source_file"])
    rec = _records(loser_file)[action["loser_id"]]
    assert rec["expired_at"], result
    assert rec["expire_reason"], "a superseded mark with no reason is unreadable"
    assert not rec.get("invalid_at"), (
        f"a write-order cleanup wrote contradiction-loser semantics: {rec}")
    assert "conflicts_with" not in rec, (
        f"`expired_at` is not a contradiction, so a trace on it would claim one: {rec}")
    # The winner of a write-order pair is the NEWER claim, which stays as it was.
    winner = _records(root / Path(action["winner_source_file"]))[action["winner_id"]]
    assert not winner.get("expired_at") and not winner.get("invalid_at"), winner
    assert "conflicts_with" not in winner, winner


def test_the_mark_count_reported_is_the_count_the_writer_returned(
        tmp_path, monkeypatch):
    """`expired_count` is the writer's number, carried through, not a constant.

    Asserted at the seam the number is made at: `apply_action` reports
    `_apply_fact_marks`' `marked`, and nothing else decides it. A node that only ever
    ran a one-fact action would stay green if the field became the literal 1 — which is
    the failure this file must not repeat — so the writer here is replaced by one that
    reports TWO records touched, and the action has to say 2. The inverse is pinned too:
    a writer that touched nothing yields 0, not 1.
    """
    root, action = _planned_confidence_action(tmp_path, monkeypatch)
    real = fi._apply_fact_marks

    def fake(marks, files, **kwargs):
        out = real(marks, files, **kwargs)
        # Report one record more than was really touched. The store is unchanged, so
        # the only place the extra count can come from is the writer's own answer.
        out["marked"] = out["marked"] + 1
        out["matched_facts"] = list(out["matched_facts"]) + [
            {"fact": "phantom", "file": "Idcol/Idcol-state.md", "how": "identity",
             "id": "fact-001"}]
        return out

    monkeypatch.setattr(fi, "_apply_fact_marks", fake)
    reported = fi.apply_action(dict(action), _iso(1))
    assert reported["expired_count"] == 2, (
        f"the action reported its own constant, not the writer's count: {reported}")
    # The other key that travels with it. The writer's answer here names ONE record
    # this action aimed and one it did not, so exactly one resolution happened and the
    # figure has to say one: `traces_written` is counted over what the writer touched AT
    # the aimed key, the way `_fact_resolve_apply:906-907` counts `key in traces`. A
    # looser `1 if trace is not None` reads the same here by luck, so the case below —
    # where nothing the writer touched matches the aimed key — is the one that separates
    # the two, and it is asserted in its own node.
    assert reported["traces_written"] == 1 == len(reported["traces"]), reported

    monkeypatch.setattr(fi, "_apply_fact_marks",
                        lambda marks, files, **kwargs: {
                            "marked": 0, "matched_facts": [], "unapplied": [],
                            "already_marked": [], "files_touched": [],
                            "files_scanned": []})
    quiet = fi.apply_action(dict(action), _iso(1))
    assert quiet["expired_count"] == 0 and quiet["traces_written"] == 0, quiet
    assert root.exists()


def _planned_superseded_action(tmp_path, monkeypatch):
    """`plan_entity`'s write-order action, with nothing about it restated here.

    The pair is the detector's own opposing-terms shape with confidences 0.02 apart:
    too close for `MIN_CONFIDENCE_GAP`, 37 days apart for `MIN_STALE_GAP_DAYS`, which is
    exactly the ground the planner's `_age_basis` stands on. Separate entity so this
    node's store cannot be the one a confidence node is reading.
    """
    root, st = _fresh_tree(tmp_path, monkeypatch)
    _write(root, [
        {"file": "Aged/Aged-state.md", "id": "fact-001",
         "fact": "The indexer is disabled.", "confidence": 0.9, "age_days": 40},
        {"file": "Aged/Aged-state.md", "id": "fact-002",
         "fact": "The indexer is enabled.", "confidence": 0.9, "age_days": 3},
    ])
    st.facts_idx.reindex(root=root)
    planned = fi.plan_entity("Aged")
    action = next((a for a in planned["actions"] if a["kind"] == "superseded"), None)
    assert action is not None, (
        "the fixture did not reach the planner's write-order branch; a plan of "
        f"{[a['kind'] for a in planned['actions']]} would silently test nothing: "
        f"{planned['actions']}")
    return root, action


def test_an_attributed_loser_with_a_file_less_winner_marks_and_writes_no_trace(
        tmp_path, monkeypatch):
    """The veto on its own: the loser CAN be aimed, the winner still cannot be named.

    `test_a_confidence_action_with_an_unnamed_winner…` reaches its `untraceable` case
    through an unattributed LOSER, which never gets as far as `_confidence_trace` — with
    no loser file the identity aim is off, so no trace is ever considered and deleting
    the veto changes nothing there. This is the action that actually asks the question:
    (file, id) present for the loser, so the mark goes on the identity path, and a winner
    with an id but no file.

    That is not an address. Ids are a per-file counter, so `fact-001` names a record in
    each category file of this fixture (#874 was one call invalidating 25 facts to change
    2), and `fact_identity` (`facts.py:296-300`) returns None for exactly that pair — the
    rule `_confidence_trace` copies. So the mark lands alone. Deleting the veto turns
    this node red; disabling the identity aim does not, which is what makes the two
    guards separately pinned.
    """
    action = dict(_confidence_action(), winner_source_file="")
    assert action["loser_source_file"] and action["loser_id"], action
    root, _loser, result = _apply(tmp_path, monkeypatch, action)
    assert result["expired_count"] == 1, result
    assert result["traces_written"] == 0 and result["untraceable"] == 1, result

    rec = _records(root / _LOSER_FILE)[_LOSER_ID]
    assert rec["invalid_at"], "the winner's missing file must not block the mark"
    assert "conflicts_with" not in rec, (
        f"a trace pointing at an id with no file names every fact sharing it: {rec}")


def test_a_mark_the_writer_did_not_apply_at_the_aimed_key_writes_no_trace(
        tmp_path, monkeypatch):
    """The predicate that separates a counted resolution from an intended one.

    `traces_written` must be 0 when the writer's answer contains no record at the key
    this action aimed — the `already_marked` shape, where a loser that already carries
    `invalid_at` is reported as already marked and, by the rule
    `_fact_resolve_apply:904-907` states, "gets no second trace". Counting the intended
    trace here would put a resolution into the nightly figure that no record carries,
    and `traces` (a list built from intent) would disagree with the count built from the
    write.

    Built by answering the mark with an empty `matched_facts` and a non-empty
    `already_marked`: the rerun that the live loop itself takes is stopped earlier, by
    `_aim_substring` (an invalidated fact is no longer active, `retrieval.py:231`), so
    the accounting cannot be reached that way — recorded on #1817.
    """
    root, action = _planned_confidence_action(tmp_path, monkeypatch)
    aimed = (action["loser_source_file"], action["loser_id"])

    def already(marks, files, **kwargs):
        assert aimed in marks, f"the action did not aim the key it carries: {marks}"
        return {"marked": 0, "matched_facts": [], "unapplied": [],
                "already_marked": [{"fact": action["loser_fact"],
                                    "file": aimed[0], "how": "identity",
                                    "id": aimed[1]}],
                "files_touched": [], "files_scanned": []}

    monkeypatch.setattr(fi, "_apply_fact_marks", already)
    result = fi.apply_action(dict(action), _iso(1))
    assert result["expired_count"] == 0, result
    assert result["traces_written"] == 0, (
        f"a resolution no record carries was counted: {result}")
    assert result["traces"] == [] and result["untraceable"] == 0, result
    assert not [r for path in sorted(root.rglob("*.md"))
                for r in _records(path).values() if r.get("conflicts_with")], (
        "the writer applied nothing at the aimed key, so no file may carry a trace")
