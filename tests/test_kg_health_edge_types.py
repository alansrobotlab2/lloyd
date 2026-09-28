"""kg_health.py's `edges.by_type` — one key per relation (#1161 clause 4).

The 2026-09-15 snapshot that surfaced this item listed `related_to` 5,046 and
`related-to` 49 as two relations, because `by_type` counted the raw `type`
column. A histogram that splits one relation in two makes the vocabulary look
bigger than it is and lets any per-type rule silently see half of it. The
snapshot is a file a person reads and a metric later reports quote, so the fold
belongs before the count, not in whoever opens the JSON afterwards.
"""
import importlib.util
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
    """Clause 4, second fixture: one type under the floor, no type over the share limit.

    `conflicts_with` at 1 over 101 active edges is 2026-09-26's measurement modulo
    the store's size — `mentions` there held 76.5%, but this fixture deliberately
    spreads the rest so share is *not* failing (40 of 101 = 39.6%, under the 50%
    limit) and the under-floor type is the only drift. The attribution must name the
    type, so the line answers "which type, and because of which rule" instead of
    leaving the count next to an unattributed FAIL.
    """
    dist = {"mentions": 40, "uses": 35, "related_to": 25, "conflicts_with": 1}
    line = khr.edge_type_cardinality(dist)

    assert line.startswith("Edge-type cardinality: FAIL — "), line
    assert "conditions failing: types below the 5-use floor (conflicts_with (1))" in line, line
    assert "dominant type share" not in _failing_segment(line), \
        f"a 39.6% leader is not over the 50% limit: {_failing_segment(line)}"


def test_both_conditions_are_attributed_together_and_a_clean_vocabulary_names_none():
    """Attribution lists every failing condition, and PASS names none.

    The joint row (mentions 60 of 101 = 59.4% over the limit, `conflicts_with` at 1
    under it) is the shape tomorrow's store has if `describes (1)` survives into it
    alongside `conflicts_with (1)`; naming only the first would let a fix for one
    condition hide the other. The clean row keeps the PASS wording
    `test_cardinality_passes_a_spread_vocabulary_with_no_rare_type` pins.
    """
    # `previous` carries the share condition into this fixture: 59.4% is under the
    # 80% ceiling, so the joint row needs a lower previous night to be a joint row.
    both = khr.edge_type_cardinality(
        {"mentions": 60, "uses": 35, "related_to": 5, "conflicts_with": 1},
        previous={"date": "2026-09-27", "share": 0.58})
    failing_both = _failing_segment(both)
    assert failing_both.startswith("types below the 5-use floor (conflicts_with (1))"), both
    assert ("dominant type share (mentions 59.4% > previous night 2026-09-27 "
            "58.0%)") in failing_both, both

    clean = khr.edge_type_cardinality({"uses": 40, "part_of": 35, "related_to": 25})
    assert clean.startswith("Edge-type cardinality: PASS — "), clean
    assert _failing_segment(clean) == "none\n" or _failing_segment(clean) == "none", clean


# ── #1658: a grace for new types, and a share that is measured as a trend ──────
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
# The tests below pin the two new rules AND the floor under them: the grace defers
# (a 30-day-old singleton still fails), the ceiling still catches a runaway leader on
# its own with no history to compare against, a rising share still fails, and a PASS
# still prints the measured share so "it passed" is never mistaken for "it stopped
# measuring".
NOW_1658 = dt.datetime(2026, 9, 28, 12, tzinfo=dt.timezone.utc)
CONFLICTS_EDGE_2026_09_26 = "2026-09-26T05:11:43Z"     # edge id 44916, origin fact_relate
NIGHT_BEFORE = {"date": "2026-09-27", "share": 0.765}  # 09-27 measured: 25,849/36,959 etc.


def test_a_type_younger_than_the_grace_is_exempt_from_the_floor_and_named_as_in_grace():
    """#1658 clause 1, both halves. `conflicts_with` at 1 of 101 active edges, first
    seen 2 days ago (its real date) → PASS, and it is named as in grace so the
    exemption is visible rather than silent. First seen 30 days ago → FAIL naming
    `types below the 5-use floor (conflicts_with (1))`, because a grace defers the
    floor and does not remove it: these two singletons will not grow, so the line
    re-catches them when the window closes and a human rules on the permanent floor.
    """
    dist = {"mentions": 40, "uses": 35, "related_to": 25, "conflicts_with": 1}
    fresh = khr.edge_type_cardinality(
        dist, first_seen={"conflicts_with": CONFLICTS_EDGE_2026_09_26},
        previous=NIGHT_BEFORE, now=NOW_1658)
    assert fresh.startswith("Edge-type cardinality: PASS — "), fresh
    assert "in grace" in fresh and "conflicts_with (1)" in fresh, fresh
    assert "conditions failing: none" in fresh, fresh
    # The exemption has to be stated as an exemption, not as a clean floor: PASS
    # with `conflicts_with` simply absent would read as "no type is under the floor".
    assert khr.EDGE_TYPE_GRACE_NIGHTS == 14, khr.EDGE_TYPE_GRACE_NIGHTS

    stale = khr.edge_type_cardinality(
        dist, first_seen={"conflicts_with": "2026-08-29T05:11:43Z"},
        previous=NIGHT_BEFORE, now=NOW_1658)
    assert stale.startswith("Edge-type cardinality: FAIL — "), stale
    assert ("conditions failing: types below the 5-use floor "
            "(conflicts_with (1))") in stale, stale
    assert "in grace" not in stale, stale


def test_the_share_fails_on_the_ceiling_or_a_rise_and_passes_a_falling_share():
    """#1658 clause 2: the dominant-type condition is a trend with a level under it.

    70.0% fallen from 76.5% → PASS (this is 2026-09-26/27's actual movement, and the
    old bound FAILed it nightly). 70.0% risen from 69.8% → FAIL naming the dominant
    type share, which is the case the old bound could not express: a store drifting
    the wrong way while sitting below any fixed line.
    """
    fell = khr.edge_type_cardinality({"mentions": 70, "uses": 30},
                                     previous={"date": "2026-09-27", "share": 0.765},
                                     now=NOW_1658)
    assert fell.startswith("Edge-type cardinality: PASS — "), fell
    rose = khr.edge_type_cardinality({"mentions": 70, "uses": 30},
                                     previous={"date": "2026-09-27", "share": 0.698},
                                     now=NOW_1658)
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
    line = khr.edge_type_cardinality({"mentions": 24941, "uses": 5316,
                                      "related_to": 3534},
                                     previous=prev, now=NOW_1658)
    assert line.startswith("Edge-type cardinality: PASS — "), line

    no_history = khr.edge_type_cardinality({"mentions": 70, "uses": 30}, now=NOW_1658)
    assert no_history.startswith("Edge-type cardinality: PASS — "), no_history
    assert "no previous night" in no_history, no_history
    runaway = khr.edge_type_cardinality({"mentions": 85, "uses": 15}, now=NOW_1658)
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
                                       previous={"date": "2026-09-27", "share": 0.765},
                                       now=NOW_1658)
    for fragment in ("mentions", "70.0%", "100 active edges"):
        assert fragment in passed, (fragment, passed)
    assert khr.edge_type_cardinality({"mentions": 85, "uses": 15},
                                     now=NOW_1658).startswith(
        "Edge-type cardinality: FAIL — ")


def test_the_quoted_bound_is_the_one_that_can_fail_the_line_and_grace_lists_only_rare_types():
    """#1658 clause 5, both halves.

    The retired wording printed `(limit 50%)` beside a rule that no longer uses 50,
    which is worse than printing nothing: a reader reconciling "76.5% … (limit 50%)"
    against a PASS concludes the gate is broken and stops trusting the section. So the
    line quotes the ceiling that can actually fail it, and the word "limit" is gone
    from it entirely. Second half: a type AT the floor is never decorated as in
    grace — `related_to` at exactly the minimum is not exempt from anything, and a
    grace note attached to a passing type would make the exemption look arbitrary.
    """
    line = khr.edge_type_cardinality(
        {"mentions": 40, "uses": 35, "related_to": 25, "conflicts_with": 1},
        first_seen={"conflicts_with": CONFLICTS_EDGE_2026_09_26,
                    "mentions": "2026-01-01T00:00:00Z"},
        previous=NIGHT_BEFORE, now=NOW_1658)
    assert "(limit 50%)" not in line and "limit" not in line, line
    assert f"ceiling {khr.EDGE_TYPE_MAX_SHARE:.0%}" in line, line
    grace_part = line.split("in grace", 1)[1].split(";")[0]
    assert "conflicts_with" in grace_part, line
    assert "related_to" not in grace_part and "mentions" not in grace_part, grace_part
    assert "related_to" not in line.split("types under 5 uses")[1].split(";")[0], line
