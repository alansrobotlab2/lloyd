"""kg_health.py's `edges.by_type` — one key per relation (#1161 clause 4).

The 2026-09-15 snapshot that surfaced this item listed `related_to` 5,046 and
`related-to` 49 as two relations, because `by_type` counted the raw `type`
column. A histogram that splits one relation in two makes the vocabulary look
bigger than it is and lets any per-type rule silently see half of it. The
snapshot is a file a person reads and a metric later reports quote, so the fold
belongs before the count, not in whoever opens the JSON afterwards.
"""
import importlib.util
import inspect
import json
import sys
import datetime as dt
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "kg_health_types", ROOT / "scripts" / "memory" / "kg_health.py")
kg_health = importlib.util.module_from_spec(_spec)
sys.modules["kg_health_types"] = kg_health
_spec.loader.exec_module(kg_health)

from app import kg_store  # noqa: E402

# The verdict line lives in the *report*, not the snapshot module, and is loaded the
# same importlib way `tests/test_knowledge_health_stale_facts.py` loads it — the file
# name has a hyphen, so it is not importable by module path.
_khr_spec = importlib.util.spec_from_file_location(
    "khr_cardinality", ROOT / "scripts" / "memory" / "knowledge-health-report.py")
khr = importlib.util.module_from_spec(_khr_spec)
sys.modules["khr_cardinality"] = khr
_khr_spec.loader.exec_module(khr)


def _kg_hygiene():
    """The same module object `kg_health._hygiene_section` imports, by the same
    path, so a baseline written here is the one the snapshot reads."""
    sys.path.insert(0, str(ROOT / "scripts" / "memory"))
    import kg_hygiene
    return kg_hygiene


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A real store and a facts root with one entity directory, so
    `build_snapshot` runs end to end over data this test owns."""
    facts = tmp_path / "facts"
    (facts / "vllm").mkdir(parents=True)
    monkeypatch.setattr(kg_health, "VAULT_FACTS_ROOT", facts)
    s = kg_store.configure(tmp_path / "kg.sqlite")
    yield s
    kg_store.reset()


def _legacy_row(s, source, target, typ):
    """A row in the spelling the pre-fold store wrote. `edges.add` folds now, so
    a direct INSERT is the only way to hand the snapshot a non-canonical type —
    which is exactly the case clause 4 is about."""
    s.conn.execute(
        "INSERT INTO edges(source, target, type, confidence, provenance,"
        " created_at, origin) VALUES (?,?,?,0.9,'INFERRED',"
        " '2026-09-18T08:38:03+00:00', 'conversation')",
        (source, target, typ))
    s.conn.commit()


def test_by_type_reports_each_relation_under_one_key(db):
    """Clause 4: no snapshot may list `related_to` and `related-to` side by
    side. Both rows are the same relation, so the histogram holds one key."""
    _legacy_row(db, "vllm", "Ray", "related-to")
    db.edges.add({"source": "vllm", "target": "Kubernetes", "type": "related_to",
                  "confidence": 0.9}, origin="test")

    snap = kg_health.build_snapshot()
    assert snap["edges"]["by_type"] == {"related_to": 2}, snap["edges"]["by_type"]
    assert snap["edges"]["count"] == 2
    assert [t for t in snap["edges"]["by_type"] if "-" in t] == []


def test_by_type_folds_a_spelling_the_store_never_recorded(db):
    """The canonicalisation is a rule over the type, not a table of the two
    spellings that happened to be live in September: `depends-on` folds, and a
    name in its canonical form is passed through untouched."""
    _legacy_row(db, "vllm", "Ray", "depends-on")
    _legacy_row(db, "vllm", "SGLang", "requires")

    assert kg_health.build_snapshot()["edges"]["by_type"] == {
        "depends_on": 1, "requires": 1}


def test_by_type_does_not_merge_two_relations_that_differ_by_more_than_a_hyphen(db):
    """`conflicts_with` and `competes_with` are two relations (#546 owns whether
    they should be one), so the histogram keeps two keys even though the first
    arrived hyphenated."""
    _legacy_row(db, "vllm", "SGLang", "conflicts-with")
    _legacy_row(db, "vllm", "TensorRT-LLM", "competes_with")

    assert kg_health.build_snapshot()["edges"]["by_type"] == {
        "conflicts_with": 1, "competes_with": 1}


def test_captured_at_is_utc_with_an_offset(db):
    """#822: the stamp was naive local time, so a 20:58 PDT run read as the
    previous UTC day's snapshot and no reader could tell which clock it was."""
    import datetime as dt
    stamp = dt.datetime.fromisoformat(kg_health.build_snapshot()["captured_at"])
    assert stamp.utcoffset() == dt.timedelta(0), stamp
    assert abs((dt.datetime.now(dt.timezone.utc) - stamp).total_seconds()) < 300


# ── #1535: the hygiene block passes the regrowth reference through ────────────

REGROWTH_KEYS = {"days", "new_dirs", "near_dup_new", "by_tier", "samples",
                 "skipped_vanished",                                    # every key already on disk
                 "dirs_at_baseline", "baseline_at", "baseline_path", "no_baseline_reason"}


def test_hygiene_regrowth_carries_its_baseline_reference_through_the_snapshot(db, tmp_path):
    """#1535 clause 3: the section names the reference it diffed against —
    `dirs_at_baseline` at `baseline_at` — and the health block passes the new
    keys through untouched. `_hygiene_section` filters the top level of
    kg_hygiene's snapshot and reshapes nothing else; a key dropped in there is a
    key no reader of the JSON will ever see again."""
    import datetime as dt
    kg_hygiene = _kg_hygiene()
    facts = kg_health.VAULT_FACTS_ROOT                 # the fixture's one-dir tree ("vllm")
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(facts, base)
    (facts / "VLLM").mkdir()                           # its case twin, created afterwards

    r = kg_health.build_snapshot(baseline_path=base)["hygiene"]["regrowth"]
    assert set(r) == REGROWTH_KEYS, sorted(r)
    assert r == kg_hygiene.regrowth(facts, 7, baseline_path=base)   # passed through unchanged
    assert r["days"] == 7
    assert (r["new_dirs"], r["near_dup_new"]) == (1, 1)
    assert r["samples"] == ["VLLM"]
    assert r["dirs_at_baseline"] == 1                  # the denominator, from the same walk
    assert r["no_baseline_reason"] is None
    assert dt.datetime.fromisoformat(r["baseline_at"]).utcoffset() == dt.timedelta(0)


def test_hygiene_regrowth_reports_none_rather_than_the_store_size(db, tmp_path):
    """With no baseline the snapshot carries None and the reason, which is the
    one state the old check can no longer be reproduced in: this tree has one
    entity directory, and before #1535 that is precisely where the snapshot
    printed `new_dirs: 1` beside `entities.count: 1` because a 7-day window over
    rebuilt `created_at` stamps covers any tree built this week."""
    r = kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")["hygiene"]["regrowth"]
    assert r["new_dirs"] is None and r["near_dup_new"] is None
    assert "absent.json" in r["no_baseline_reason"]
    assert r["dirs_at_baseline"] is None
    assert kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")["entities"]["count"] == 1


def test_the_summary_prints_the_reference_beside_the_regrowth_number(db, tmp_path, capsys):
    """kg_health's own line, the surface the item named: it used to read
    `near-dup regrowth 7d  51 of 12027 new dirs`, where 12,027 was the whole
    store. The window is gone from the line because the number is no longer a
    window."""
    kg_hygiene = _kg_hygiene()
    facts = kg_health.VAULT_FACTS_ROOT
    base = tmp_path / "baseline.json"
    kg_hygiene.write_baseline(facts, base)
    (facts / "VLLM").mkdir()

    kg_health.print_summary(kg_health.build_snapshot(baseline_path=base))
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "regrowth" in ln]
    assert len(lines) == 1, lines
    assert "1 of 1 new dirs" in lines[0], lines[0]
    assert "baseline" in lines[0] and "over 1 dirs" in lines[0], lines[0]
    assert "7d" not in lines[0], lines[0]


def test_the_summary_renders_a_pre_fix_snapshot_without_a_bare_count(db, tmp_path, capsys):
    """The snapshots already on disk under `_pipeline/metrics/` have no baseline
    keys and must still render — with their number labelled as having no
    reference, rather than as the growth of 11,959 directories in a week."""
    snap = kg_health.build_snapshot(baseline_path=tmp_path / "absent.json")
    snap["hygiene"]["regrowth"] = {"days": 7, "new_dirs": 11959, "near_dup_new": 51,
                                   "by_tier": {"CASE": 51}, "samples": [], "skipped_vanished": 0}

    kg_health.print_summary(snap)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "regrowth" in ln]
    assert len(lines) == 1, lines
    assert "51 of 11,959 new dirs" in lines[0], lines[0]
    assert "no baseline recorded" in lines[0], lines[0]


# ── The cardinality line names the condition that failed it (#1544 clause 4) ─────

#: What the line printed itself on the two nights this item was measured. Both FAIL,
#: and from the printed text a reader cannot tell which condition failed either one:
#: 2026-09-25 had no under-floor type at all, and 2026-09-26 had one.
LIVE_2026_09_25 = "Edge-type cardinality: FAIL — types under 5 uses: none; dominant type " \
                  "mentions is 84.7% of 35,849 active edges (limit 50%)"
LIVE_2026_09_26 = "Edge-type cardinality: FAIL — types under 5 uses: conflicts_with (1); " \
                  "dominant type mentions is 76.5% of 36,119 active edges (limit 50%)"


def _failing_segment(line: str) -> str:
    """Everything after the `conditions failing: ` marker, so an assertion about the
    attribution cannot be satisfied by the operand columns before it."""
    return line.split("conditions failing: ", 1)[1]


def test_a_share_only_failure_is_attributed_to_share_and_not_to_an_under_floor_type():
    """Clause 4, first fixture: no under-floor type, dominant type over the limit.

    `mentions` 60 of 100 with `uses` 40 is the live shape (2026-09-25: mentions at
    84.7% of 35,849 active edges and nothing under the floor), and it is the case
    the guard's own `test_either_signal_alone_fails` already calls "a catch-all with
    no rare types". The verdict must stay FAIL on the share condition alone while
    saying so: a reader who sees only `FAIL` goes looking for the rare type that is
    not there, which is what happened when `conflicts_with (1)` appeared on a line
    that share had been failing since at least the night before.
    """
    # `previous` is what makes the share condition fire here at all: 60% of the
    # store is under the 80% ceiling, and #1658 made the share a trend, so a share
    # only fails the share condition when it is ABOVE the previous night's. That is
    # the sharper trigger the old level replaced, and the attribution still has to
    # name it rather than leave a bare FAIL.
    line = khr.edge_type_cardinality(
        {"mentions": 60, "uses": 40},
        previous={"date": "2026-09-27", "share": 0.49})

    assert line.startswith("Edge-type cardinality: FAIL — "), line
    assert "types under 5 uses: none" in line, \
        "the under-floor condition must still read clean: " + line
    assert ("conditions failing: dominant type share (mentions 60.0% > previous "
            "night 2026-09-27 49.0%)") in line, line
    assert "floor" not in _failing_segment(line), \
        f"share alone must not be attributed to the floor: {_failing_segment(line)}"


def test_an_under_floor_type_is_named_as_its_own_condition():
    """Clause 4 (#1544), second fixture, re-parameterised by #1820: one type under the
    floor, no type over the share limit.

    This test used `conflicts_with (1)` as its failing example, which is the very type
    #1820 rules on: `conflicts_with` is in `app.kg_store.EDGE_TYPES`, so a canonical
    singleton is now named and never fails. The attribution property is unchanged, so
    the fixture moves to `ships` — off-vocabulary, one edge, the type #1820 names and
    the shape `conversation_relations.py:1119` can mint. `mentions` still holds 40 of
    101 (39.6%, under the ceiling) so the under-floor type is the only drift, and the
    attribution must still name it, so the line answers "which type, and because of
    which rule" instead of leaving a count next to an unattributed FAIL.
    """
    assert "ships" not in kg_store.EDGE_TYPES, "the fixture stopped being off-vocabulary"
    dist = {"mentions": 40, "uses": 35, "related_to": 25, "ships": 1}
    line = khr.edge_type_cardinality(dist)

    assert line.startswith("Edge-type cardinality: FAIL — "), line
    assert "conditions failing: types below the 5-use floor (ships (1))" in line, line
    assert "dominant type share" not in _failing_segment(line), \
        f"a 39.6% leader is not over the ceiling: {_failing_segment(line)}"


def test_both_conditions_are_attributed_together_and_a_clean_vocabulary_names_none():
    """Attribution lists every failing condition, and PASS names none (#1544 clause 4),
    with the under-floor half of the joint fixture moved off-vocabulary by #1820.

    The joint row (mentions 60 of 101 = 59.4% over the limit, `ships` at 1 under it) is
    the shape tomorrow's store has if the classifier mints a junk type while the
    dominant type is still drifting; naming only the first would let a fix for one
    condition hide the other. It cannot be `conflicts_with` any more — canonical now
    means named-and-not-failing, which is the other test's subject. The clean row keeps
    the PASS wording `test_cardinality_passes_a_spread_vocabulary_with_no_rare_type`
    pins.
    """
    # `previous` carries the share condition into this fixture: 59.4% is under the
    # 80% ceiling, so the joint row needs a lower previous night to be a joint row.
    both = khr.edge_type_cardinality(
        {"mentions": 60, "uses": 35, "related_to": 5, "ships": 1},
        previous={"date": "2026-09-27", "share": 0.58})
    failing_both = _failing_segment(both)
    assert failing_both.startswith("types below the 5-use floor (ships (1))"), both
    assert ("dominant type share (mentions 59.4% > previous night 2026-09-27 "
            "58.0%)") in failing_both, both

    clean = khr.edge_type_cardinality({"uses": 40, "part_of": 35, "related_to": 25})
    assert clean.startswith("Edge-type cardinality: PASS — "), clean
    assert _failing_segment(clean) == "none\n" or _failing_segment(clean) == "none", clean


# ── #1658: a share measured as a trend; #1820: the floor reads vocabulary ───────
#
# Both halves of this gate were reporting a condition no reader could act on. The
# floor caught `conflicts_with (1)` and `describes (1)` — one edge each, both minted
# 2026-09-26 by `fact_relate`, which REFUSES a type outside the canonical `EDGE_TYPES`
# set, so the store's only writer had used an approved type once and was told it was
# a one-off. And the share bound of 50% FAILed every night against a number that was
# falling: 09-23 92.2%, 09-24 83.9%, 09-25 84.6%, 09-26 76.3%, 09-27 69.4-69.8%, with
# task #74 retyping ~2.5k `mentions` edges a night. A verdict that never changes
# carries no information, so the reader stops reading the section — which is how a
# real catch-all would get through.
#
# #1658 answered the first half with a 14-night grace, which only deferred it: those
# two singletons do not grow, so the line would re-catch them as the window closed on
# 2026-10-10. #1820 replaced the age split with the distinction the grace could not
# draw — an under-floor type OUTSIDE `app.kg_store.EDGE_TYPES` fails at any age, a
# canonical one is named and never fails — and #1931 then dropped the count gate from
# that check entirely, so an outside-vocabulary name fails at any COUNT too and the
# floor only chooses which failing condition names it (pinned at the end of this file).
# The ceiling-and-trend half of #1658 is untouched, still pinned below: the ceiling catches a runaway leader on
# its own with no history to compare against, a rising share still fails, and a PASS
# still prints the measured share so "it passed" is never mistaken for "it stopped
# measuring".
NOW_1658 = dt.datetime(2026, 9, 28, 12, tzinfo=dt.timezone.utc)
NIGHT_BEFORE = {"date": "2026-09-27", "share": 0.765}  # 09-27 measured: 25,849/36,959 etc.

#: Everything #1658's grace needed, by the names it had in the report. None of it
#: survives #1820: the floor reads the store's own type vocabulary, so nobody reads a
#: type's age, and a helper whose only caller is gone is not kept around in case.
AGE_MACHINERY = ("EDGE_TYPE_GRACE_NIGHTS", "first_seen_by_type", "_nights_old")


def test_the_under_floor_split_reads_vocabulary_and_no_age_machinery_survives():
    """#1658 clause 1, re-parameterised to #1820's rule (#1820 clause 4): what keeps an
    under-floor type out of the FAIL is that its type is in `app.kg_store.EDGE_TYPES`,
    not that it is young — and the age machinery is deleted, not left in place unused.

    The names are pinned absent from the report's own source rather than from its
    namespace, because `hasattr` could not catch two of the three: `_nights_old` is a
    closure inside `edge_type_cardinality` and was never a module attribute, and
    `"in grace"` is a literal inside an f-string. A surviving copy of either would
    still be a grace the code could fall back to, and the live line is only guaranteed
    to stop saying `in grace` (owed-check 1) if the literal is gone.
    """
    src = (ROOT / "scripts" / "memory" / "knowledge-health-report.py").read_text(
        encoding="utf-8")
    for gone in (*AGE_MACHINERY, "in grace", "first_seen"):
        assert gone not in src, f"{gone} still appears in the report"
    # The behaviour behind the name: age is not an input at all, so no caller — the
    # nightly report, a test, or tomorrow's repair script — can exempt a young type.
    assert "first_seen" not in inspect.signature(khr.edge_type_cardinality).parameters


def test_an_off_vocabulary_type_under_the_floor_fails_at_any_age_and_names_its_count():
    """#1820 clause 1: the case the floor exists for, and the age cannot excuse it.

    `ships` is one edge of a type outside `app.kg_store.EDGE_TYPES` — what the Stage-2
    classifier proposes (`conversation_relations.py:1119`) or a rebuild payload carries
    (`kg_rebuild.py:597`), neither of which consults the vocabulary and neither of which
    the store rejects: `_Edges.add` only canonicalises the name it is given, and the
    refusal lives in `_fact_relate` alone (`agent_mcp/facts.py:1025`). Nothing else in
    the system reads the stored rows against the vocabulary — `tests/
    test_edge_type_vocabulary.py:48-56` pins only that the DECLARED sets agree with
    `EDGE_TYPES` — so this line is where such a type surfaces. It must fail on the night
    it appears: the store was carrying `ships`, `informs`, `co_accessed`,
    `upgrade_candidate_for` and `shares_mechanism_with` live over 53,902 edges as
    recently as 2026-09-19 (#546's autotriage), so this is not a hypothetical shape.

    Two shapes, both failing, because the clause asks for both: the dist with no age
    information supplied at all, and the same dist arrived at the way the nightly report
    does — from edge rows whose `created_at` is one day old, through
    `compute_relationship_stats`, which is the seam a young type used to reach the gate.
    `first_seen` is gone from the signature, so an age cannot be offered even by a
    caller that still wanted to.
    """
    assert "ships" not in kg_store.EDGE_TYPES, "the fixture stopped being off-vocabulary"
    dist = {"mentions": 40, "uses": 35, "related_to": 25, "ships": 1}

    line = khr.edge_type_cardinality(dist)
    assert line.startswith("Edge-type cardinality: FAIL — "), line
    assert "ships (1)" in line, line
    assert "conditions failing: types below the 5-use floor (ships (1))" in line, line

    young = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    rows = ([{"source": f"S{i}", "target": f"T{i}", "type": "mentions",
              "created_at": young.isoformat()} for i in range(40)]
            + [{"source": f"U{i}", "target": f"V{i}", "type": "uses",
                "created_at": young.isoformat()} for i in range(35)]
            + [{"source": f"R{i}", "target": f"W{i}", "type": "related_to",
                "created_at": young.isoformat()} for i in range(25)]
            + [{"source": "A", "target": "B", "type": "ships",
                "created_at": young.isoformat()}])
    from_store = khr.edge_type_cardinality(
        khr.compute_relationship_stats(rows, {})["type_distribution"])
    assert from_store.startswith("Edge-type cardinality: FAIL — "), from_store
    assert "ships (1)" in from_store, from_store
    with pytest.raises(TypeError):
        khr.edge_type_cardinality(dist, first_seen={"ships": young.isoformat()})


def test_a_canonical_type_under_the_floor_is_named_and_never_fails_the_line():
    """#1820 clause 2: `fact_relate` refuses a type outside `EDGE_TYPES`, so a
    canonical singleton is a legitimate first use and the line may only report it.

    The distribution is 2026-09-29's live shape reduced to a testable size: two
    approved types at one edge each (`conflicts_with` is edge 44916 and `describes` is
    48187, both `expired_at` NULL on the live store) over a spread leader that is not
    itself failing the share half. Before today this PASSed only because both were
    inside the 14-night window and printed `in grace`; on 2026-10-10 the window closed
    and the same two rows FAILed. Now the ruling is the one the row itself supports —
    `fact_relate` would not have accepted a type outside `EDGE_TYPES`, so the store
    could not have made these by accident — and the exemption survives the date because
    it never depended on one. Both singletons stay NAMED on the line: reporting a rare
    type and staying silent about it are the same loss of information in opposite
    directions, and this line is the only place either appears.
    """
    for approved in ("conflicts_with", "describes"):
        assert approved in kg_store.EDGE_TYPES, f"{approved} left the vocabulary"
    line = khr.edge_type_cardinality({"mentions": 40, "uses": 35, "related_to": 25,
                                      "conflicts_with": 1, "describes": 1})
    assert line.startswith("Edge-type cardinality: PASS — "), line
    under_floor_part = line.split("types under 5 uses", 1)[1].split(";")[0]
    assert "conflicts_with (1)" in under_floor_part, line
    assert "describes (1)" in under_floor_part, line
    assert "conditions failing: none" in line, line
    assert "in grace" not in line, line


def test_the_share_fails_on_the_ceiling_or_a_rise_and_passes_a_falling_share():
    """#1658 clause 2: the dominant-type condition is a trend with a level under it.

    70.0% fallen from 76.5% → PASS (this is 2026-09-26/27's actual movement, and the
    old bound FAILed it nightly). 70.0% risen from 69.8% → FAIL naming the dominant
    type share, which is the case the old bound could not express: a store drifting
    the wrong way while sitting below any fixed line.
    """
    fell = khr.edge_type_cardinality({"mentions": 70, "uses": 30},
                                     previous={"date": "2026-09-27", "share": 0.765})
    assert fell.startswith("Edge-type cardinality: PASS — "), fell
    rose = khr.edge_type_cardinality({"mentions": 70, "uses": 30},
                                     previous={"date": "2026-09-27", "share": 0.698})
    assert rose.startswith("Edge-type cardinality: FAIL — "), rose
    assert ("dominant type share (mentions 70.0% > previous night 2026-09-27 "
            "69.8%)") in _failing_segment(rose), rose


def test_the_previous_night_is_day_aligned_and_no_history_only_lets_the_ceiling_fail(tmp_path):
    """#1658 clause 3, both halves.

    Day-aligned, not snapshot-to-snapshot: `kg_health` runs 3-5× a day and `mentions`
    grew between CONSECUTIVE snapshots in every hour measured on 2026-09-27 (+149,
    +152, +152, +84, +68), so a snapshot-to-snapshot comparison would report "rising"
    on every single nightly run — the same permanent-FAIL failure mode the 50% bound
    had. The fixture therefore hands over a snapshot count that ROSE since the last
    snapshot (24,941 from 24,505) while the dated share FELL (65.83% → 63.4%), and
    the verdict must be PASS: what the ruling asks about is the overnight trend, not
    the intra-day churn.

    And with no snapshot directory at all, only the ceiling can fail, and the line
    says no previous night was available rather than implying it compared and won.
    """
    empty = tmp_path / "metrics"
    assert khr.previous_dated_share(NOW_1658, empty) is None

    d = empty
    d.mkdir(parents=True)
    # Last night's final snapshot: `mentions` 24,505 of 31,005 = 79.0%. Today's count
    # is HIGHER (24,941) because the store grows ~700 edges a night, and its SHARE is
    # lower (73.8%) because task #74 retypes ~2.5k `mentions` edges into `uses` and
    # `related_to` — count up, share down, which is exactly the pair a snapshot-to-
    # snapshot reader would call "rising" and a day-aligned one calls what it is.
    (d / "kg-health-2026-09-27T223448Z.json").write_text(json.dumps({
        "captured_at": "2026-09-27T22:34:48.000+00:00",
        "edges": {"by_type": {"mentions": 24505, "uses": 3000, "related_to": 3500}}}),
        encoding="utf-8")
    prev = khr.previous_dated_share(NOW_1658, d)
    assert prev and prev["date"] == "2026-09-27", prev
    assert prev["share"] == pytest.approx(24505 / 31005), prev

    today = d / "kg-health-2026-09-28T113356Z.json"
    today.write_text(json.dumps({
        "captured_at": "2026-09-28T11:33:56.000+00:00",
        "edges": {"by_type": {"mentions": 24941, "uses": 5316, "related_to": 3534}}}),
        encoding="utf-8")
    # Today's own snapshot must not become "the previous night" — that is the whole
    # day-alignment rule, and failing to apply it would FAIL the run against itself.
    assert khr.previous_dated_share(NOW_1658, d)["date"] == "2026-09-27", d
    count_rose = 24941 > 24505
    share_fell = (24941 / 33791) < prev["share"]
    assert count_rose and share_fell, "fixture no longer shows the trap"
    # `now=` is gone from every `edge_type_cardinality` call in this file: #1820 deleted
    # the only thing that read a clock (the grace window). Each assertion below is
    # #1658's, unchanged, and `NOW_1658` still times `previous_dated_share` above.
    line = khr.edge_type_cardinality({"mentions": 24941, "uses": 5316,
                                      "related_to": 3534},
                                     previous=prev)
    assert line.startswith("Edge-type cardinality: PASS — "), line

    no_history = khr.edge_type_cardinality({"mentions": 70, "uses": 30})
    assert no_history.startswith("Edge-type cardinality: PASS — "), no_history
    assert "no previous night" in no_history, no_history
    runaway = khr.edge_type_cardinality({"mentions": 85, "uses": 15})
    assert runaway.startswith("Edge-type cardinality: FAIL — "), runaway
    assert "dominant type share" in _failing_segment(runaway), runaway


def test_a_pass_still_prints_the_dominant_type_its_share_and_the_active_total():
    """#1658 clause 4: the purpose survived the softening, so prove it on the PASS.

    A gate that stops failing is only an improvement if it still measures. Asserted
    on the PASSING line — the FAIL line was always informative, the PASS line is the
    one that can quietly become the word "PASS" and nothing else. And a leader at
    85% still fails with no history to compare against: the ceiling is the catch-all
    the whole section exists for.
    """
    passed = khr.edge_type_cardinality({"mentions": 70, "uses": 30},
                                       previous={"date": "2026-09-27", "share": 0.765})
    for fragment in ("mentions", "70.0%", "100 active edges"):
        assert fragment in passed, (fragment, passed)
    assert khr.edge_type_cardinality({"mentions": 85, "uses": 15}).startswith(
        "Edge-type cardinality: FAIL — ")


def test_the_quoted_bound_is_the_one_that_can_fail_the_line_and_the_floor_lists_only_rare_types():
    """#1658 clause 5, both halves, with the second half re-pointed by #1820.

    The retired wording printed `(limit 50%)` beside a rule that no longer uses 50,
    which is worse than printing nothing: a reader reconciling "76.5% … (limit 50%)"
    against a PASS concludes the gate is broken and stops trusting the section. So the
    line quotes the ceiling that can actually fail it, and the word "limit" is gone
    from it entirely — unchanged by #1820, and asserted here with the grace kwarg
    removed because there is no grace left to pass. Second half: what the under-floor
    column lists is under-floor types and nothing else. `related_to` at 25 and
    `mentions` at 40 are both above the 5-use minimum, so neither may appear in the
    column, while the canonical `conflicts_with (1)` must: the column is the line's
    only place a rare type is named, and #1820 made it the only place the canonical
    ones are named at all.
    """
    line = khr.edge_type_cardinality(
        {"mentions": 40, "uses": 35, "related_to": 25, "conflicts_with": 1},
        previous=NIGHT_BEFORE)
    assert "(limit 50%)" not in line and "limit" not in line, line
    assert f"ceiling {khr.EDGE_TYPE_MAX_SHARE:.0%}" in line, line
    under_floor_part = line.split("types under 5 uses", 1)[1].split(";")[0]
    assert "conflicts_with (1)" in under_floor_part, line
    assert "related_to" not in under_floor_part and "mentions" not in under_floor_part, \
        under_floor_part


# ── #1931: the floor is a pure vocabulary test — off-vocabulary fails at ANY count ──
#
# #1820 moved the under-floor decision from age to vocabulary but left the count
# gating it: `under = … if n < EDGE_TYPE_MIN_USES`, vocabulary only tested inside
# that. Measured on HEAD f952852f (2026-09-30): `{mentions 40, uses 35,
# related_to 25, ships 5}` printed PASS naming nothing, and `ships: 500` failed
# only on the share ceiling — so a classifier that minted a junk name repeatedly
# read as clean, while one use of the same name failed. #1931 drops the count gate
# from the vocabulary condition; the floor now only chooses WHICH failing
# condition names an outside-vocabulary type. Membership is judged on
# `canonical_edge_type(t)` because the report's histogram counts the raw row type
# while `kg_health.py` folds before counting — without the same fold here the
# vocabulary test would fire on a pre-#1161 spelling instead of on a name.
# The share half of the line is untouched by this section.

def test_an_off_vocabulary_type_at_or_above_the_floor_fails_and_names_its_count():
    """#1931 clause 1: the hole #1820 left open, closed at both boundaries.

    `ships` at EXACTLY 5 uses (5 is the floor: `n < 5` is under it, 5 itself is
    not) and at 500 uses must both FAIL and name the count. The leader stays
    under the 80% ceiling in both fixtures — 40/105 = 38.1% and 500/1,500 of
    1,500 total is 33.3%, and no previous night is given — so the new
    `types outside the edge-type vocabulary` condition is the only thing that can
    fail these lines. `types under 5 uses:` reads `none` at exactly 5 uses: this
    is precisely the name the old code never put under the floor.
    """
    assert "ships" not in kg_store.EDGE_TYPES, "the fixture stopped being off-vocabulary"

    at_floor = khr.edge_type_cardinality(
        {"mentions": 40, "uses": 35, "related_to": 25, "ships": 5})
    assert at_floor.startswith("Edge-type cardinality: FAIL — "), at_floor
    assert "ships (5)" in at_floor, at_floor
    assert "conditions failing: types outside the edge-type vocabulary " \
           "(ships (5))" in at_floor, at_floor
    assert "types under 5 uses: none" in at_floor, at_floor

    above = khr.edge_type_cardinality(
        {"mentions": 400, "uses": 350, "related_to": 250, "ships": 500})
    assert above.startswith("Edge-type cardinality: FAIL — "), above
    assert "ships (500)" in above, above
    assert "conditions failing: types outside the edge-type vocabulary " \
           "(ships (500))" in above, above
    assert "dominant type share" not in _failing_segment(above), \
        f"33.3% is under the ceiling: {_failing_segment(above)}"


def test_a_pre_canonical_spelling_of_an_approved_type_is_never_read_as_off_vocabulary():
    """#1931 clause 2: the fold must ship with the vocabulary test, not after it.

    The report's histogram counts the RAW row type (`compute_relationship_stats`
    counts `edge.get("type", "unknown")`), while `kg_health.py` folds every
    snapshot through `canonical_edge_type` before counting. Change 1 alone would
    therefore have turned every pre-#1161 spelling of an approved relation into a
    nightly FAIL: `related-to` is not a member of `EDGE_TYPES` under its raw
    spelling — what makes it approved is that `canonical_edge_type("related-to")`
    is. Membership is now judged through the same fold the snapshot uses.

    Three legs, because the fold must hold on each verdict path: `related-to` at
    40 uses PASSes (it fails neither half); `conflicts-with` at 1 is the
    canonicalising fold of an approved type, so it is NAMED in `types under 5
    uses:` and fails nothing — the same report-not-fail #1820 ruled for canonical
    singletons; and the fold must not launder real junk: `ships-with` folds to
    `ships_with`, still outside `EDGE_TYPES`, and FAILs at one use inside the
    floor condition, exactly like its canonical twin.
    """
    assert kg_store.canonical_edge_type("related-to") == "related_to"
    assert "related-to" not in kg_store.EDGE_TYPES, "raw fixture name joined the vocabulary"
    assert "ships_with" not in kg_store.EDGE_TYPES, "folded junk name joined the vocabulary"

    passed = khr.edge_type_cardinality({"mentions": 40, "uses": 35, "related-to": 40})
    assert passed.startswith("Edge-type cardinality: PASS — "), passed
    assert "conditions failing: none" in passed, passed

    named = khr.edge_type_cardinality(
        {"mentions": 40, "uses": 35, "related_to": 25, "conflicts-with": 1})
    assert named.startswith("Edge-type cardinality: PASS — "), named
    assert "conflicts-with (1)" in named.split("conditions failing", 1)[0], named
    assert "conditions failing: none" in named, named

    junk = khr.edge_type_cardinality(
        {"mentions": 40, "uses": 35, "related_to": 25, "ships-with": 1})
    assert junk.startswith("Edge-type cardinality: FAIL — "), junk
    assert "ships-with (1)" in _failing_segment(junk), junk

    # The seam the fold actually serves: the nightly line is not handed a curated
    # dict but the histogram `compute_relationship_stats` builds from raw store
    # rows (`edge.get("type", "unknown")`, no fold). These rows carry the hyphen
    # the pre-#1161 store wrote, so the dist reaching the gate holds the RAW key —
    # this is the call that would FAIL nightly on spelling alone without the fold.
    young = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)
    rows = ([{"source": f"S{i}", "target": f"T{i}", "type": "mentions",
              "created_at": young.isoformat()} for i in range(40)]
            + [{"source": f"U{i}", "target": f"V{i}", "type": "uses",
                "created_at": young.isoformat()} for i in range(35)]
            + [{"source": f"R{i}", "target": f"W{i}", "type": "related-to",
                "created_at": young.isoformat()} for i in range(40)])
    raw_dist = khr.compute_relationship_stats(rows, {})["type_distribution"]
    assert raw_dist == {"mentions": 40, "uses": 35, "related-to": 40}, raw_dist
    from_store = khr.edge_type_cardinality(raw_dist)
    assert from_store.startswith("Edge-type cardinality: PASS — "), from_store
    assert "conditions failing: none" in from_store, from_store


def test_a_canonical_distribution_still_passes_and_still_names_its_singleton():
    """#1931 clause 3: the live shape must not move.

    The 2026-09-30 witness (`backlog/data/kg-health-2026-09-30T222931Z.json`:
    38,551 active edges, 13 types, zero names outside the vocabulary, all three
    under-floor types approved) reduced to a testable size. Dropping the count
    gate from the vocabulary condition must keep this distribution PASSing, keep
    naming `conflicts_with (1)` in the `types under 5 uses:` column, and keep
    `conditions failing: none`: reporting approved singletons without failing
    them is #1820's ruling, and the line still holds it because the FAIL side
    keys on vocabulary, not on count.
    """
    line = khr.edge_type_cardinality({"mentions": 40, "uses": 35, "related_to": 25,
                                      "conflicts_with": 1})
    assert line.startswith("Edge-type cardinality: PASS — "), line
    under_floor_part = line.split("types under 5 uses", 1)[1].split(";")[0]
    assert "conflicts_with (1)" in under_floor_part, line
    assert "conditions failing: none" in line, line


def test_the_under_floor_off_vocabulary_failing_text_is_unchanged_by_the_new_condition():
    """#1931 clause 4: the two historical failing-condition strings, byte-for-byte.

    #1544 pinned the attribution wording and #1820 pinned its off-vocabulary
    subject; #1931 may add a condition but may not retitle an existing one.
    `ships (1)` inside `types below the 5-use floor (…)` is the string
    `test_an_under_floor_type_is_named_as_its_own_condition` and the nightly
    reader quote, and `informs (1), ships (1)` is the string
    `tests/test_knowledge_health_stale_facts.py`'s drifted-store fixture quotes;
    both are re-asserted here against the real function, alongside those untouched
    nodes. An off-vocabulary type UNDER the floor stays on the floor condition
    alone — the new condition names only the at-or-above-floor types the old code
    silently passed. The mixed fixture pins the coexistence: `informs (1)` under
    the floor and `ships (10)` above it fail together, floor condition first, and
    the leader at 60/106 = 56.6% with no previous night fails neither share half.
    """
    one = khr.edge_type_cardinality({"mentions": 40, "uses": 35, "related_to": 25,
                                     "ships": 1})
    assert "conditions failing: types below the 5-use floor (ships (1))" in one, one

    both_off = khr.edge_type_cardinality(
        {"mentions": 60, "uses": 35, "depends_on": 3, "informs": 1, "ships": 1})
    assert ("conditions failing: types below the 5-use floor "
            "(informs (1), ships (1))") in both_off, both_off
    assert "depends_on" not in _failing_segment(both_off), both_off
    assert "edge-type vocabulary" not in _failing_segment(both_off), \
        f"an under-floor off-vocab type belongs to the floor condition alone: {_failing_segment(both_off)}"

    mixed = khr.edge_type_cardinality(
        {"mentions": 60, "uses": 35, "informs": 1, "ships": 10})
    assert mixed.startswith("Edge-type cardinality: FAIL — "), mixed
    failing_mixed = _failing_segment(mixed)
    assert failing_mixed.startswith("types below the 5-use floor (informs (1))"), mixed
    assert "types outside the edge-type vocabulary (ships (10))" in failing_mixed, mixed
    assert "dominant type share" not in failing_mixed, mixed
