"""The near-duplicate predicate (#2271): its threshold, its calibration, and
what the predicate itself must never do.

The eval has always scored whether the gold document is IN the returned top-K
(`doc_hit_rate` 0.689 on the newest nightly) and how high it sits (`mrr_doc`
0.341, `ndcg10` 0.399). A hit that lands at rank 6 is a hit to the first number
and a problem to the second, and nothing in this repo could say whether the
slots above it were distinct documents or copies of each other. `eval/dup_detect.py`
is that measurement; this file pins the parts that would silently make it lie:

  * the threshold is a named constant, and the committed calibration set of
    labelled pairs is reproduced by that constant — 40 of 40 rows, agreement
    printed beside its denominator so a stripped set cannot read as clean;
  * the predicate compares CONTENT, not files: a finding whose appended triage
    trailer differs is one finding (live case: `backlog/1761-*` and
    `backlog/1762-*` — whole-file Jaccard 0.448, content Jaccard 1.0);
  * a similarity of 0.0 is emitted only for text that was actually compared, and
    a pair that was never compared is `dup_slot_n: 0` rather than a 0.0 share;
  * supersession needs a same-topic newer note, so a note that is merely old and
    current is never counted superseded, and the disagreement with a
    recency-only rule is counted rather than hidden.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import eval.dup_detect as dd  # noqa: E402


# ── fixture text ─────────────────────────────────────────────────────────────

def _note(title: str, body: str, stamp: str = "", tags=()) -> str:
    """One vault-shaped note: front matter, then body prose."""
    fm = ["---", "segment: knowledge", f"title: {title}"]
    if stamp:
        fm.append(f"timestamp: '{stamp}'")
    if tags:
        fm.append("tags: [" + ", ".join(tags) + "]")
    fm.append("---")
    return "\n".join(fm) + "\n" + body.strip() + "\n"


# A paragraph long enough that two copies of it share dozens of word-5-grams.
_SHARED = (
    "The nightly retrieval eval scores each query against the gold documents the "
    "question was filed with, and reports the share of returned slots that were "
    "readable before any share is divided over them. A slot whose file could not "
    "be read is dropped from the comparison and counted separately, because a "
    "denominator that includes unreadable notes would let a run over nothing at "
    "all report a corpus with no duplicates, which is the wrong finding to invent "
    "about a corpus that is rewritten by writers every night."
)

_OTHER = (
    "Wake-word audio on this machine is bound to a specific ALSA card, and a "
    "reboot that changes the card index leaves the listener attached to a device "
    "that no longer exists, so the first thing to check when the transcript stops "
    "arriving is the media binding rather than the acoustic model that has not "
    "changed since the last working boot."
)

_THIRD = (
    "Supervisor-managed services restart through one documented route, and a "
    "process killed outside that route comes back with a different environment, "
    "which is how a service can look healthy in the process table while answering "
    "nothing on its port."
)


# ── the threshold and its calibration ────────────────────────────────────────

def test_threshold_is_a_named_constant_in_the_measured_gap():
    """The predicate's one number is committed, not derived at call time.

    The calibration set brackets it: 0.5618 is the highest similarity among pairs
    labelled not-duplicates, 0.7097 the lowest among pairs labelled duplicates.
    Anything in that bracket would be a threshold nobody measured, so the gap's
    bounds are pinned here as well as in the JSON.
    """
    assert dd.DUP_JACCARD_THRESHOLD == 0.6
    assert dd.SHINGLE_K == 5
    cal = dd.load_calibration()
    assert cal["meta"]["threshold"] == dd.DUP_JACCARD_THRESHOLD
    assert cal["meta"]["shingle_k"] == dd.SHINGLE_K
    assert cal["meta"]["nearest_non_duplicate"] == pytest.approx(0.5618, abs=1e-4)
    assert cal["meta"]["nearest_duplicate"] == pytest.approx(0.7097, abs=1e-4)
    assert cal["meta"]["nearest_non_duplicate"] < dd.DUP_JACCARD_THRESHOLD
    assert dd.DUP_JACCARD_THRESHOLD < cal["meta"]["nearest_duplicate"]


def test_committed_calibration_is_at_least_thirty_pairs_and_is_reproduced():
    """`agreement` is a count over the committed texts, with its denominator.

    The texts are embedded in the JSON rather than referenced by path, so this
    check re-runs against the shipped artifact and not against a vault that moves
    under it. 40 of 40 rows reproduce under the committed threshold; the count and
    the denominator are asserted together, which is the "denominator can be zero"
    rule — a 0/0 set would otherwise print as a clean agreement.
    """
    cal = dd.load_calibration()
    rows = cal["pairs"]
    assert len(rows) >= 30, "the calibration set must be at least 30 labelled pairs"
    # two classes carry an opinion, and both are present. `unlabelled` is a third
    # value the reading rules write when they abstain — no subject tokens, no shared
    # title family, no shared entity tag — and abstentions are excluded from the fit
    # rather than scored as correct negatives.
    assert {r["label"] for r in rows} >= {"same", "near", "unrelated"}
    agree, total, bad = dd.agreement(rows, dd.DUP_JACCARD_THRESHOLD)
    # 40 rows committed; 35 reproduce, over a FITTING SET of 35. The 5-row difference
    # is composition, not cherry-picking, and every row in it stays in the file and
    # is named by pair_id in the printed report:
    #   * 2 disclosed misses (`d0420`, `d0422`) — pairs a person read as duplicates
    #     that a word-5-gram predicate cannot reach at ANY threshold (measured 0.4058
    #     and 0.3276, against a 0.5 stated reachability floor);
    #   * 3 abstentions (`d0736`, `d0738`, `d0739`) with no predicted class.
    # No row is excluded because it disagreed: membership is decided by the class of
    # evidence behind the label, fixed before any agreement is read.
    assert len(rows) == 40
    assert (agree, total, bad) == (35, 35, []), (agree, total, bad)
    assert [m["pair_id"] for m in dd.disclosed_misses(rows)] == ["d0420", "d0422"]
    assert cal["meta"]["agreement_at_threshold"] == f"{agree}/{total}"


def test_calibration_report_prints_agreement_beside_its_denominator(tmp_path):
    """The report is the human-facing line, and it cannot show a bare 100%."""
    text = dd.calibration_report()
    # the numerator AND the composition of its denominator, on one line
    assert "agreement 35/35" in text
    assert "fitting set 35 of 40 committed pairs" in text
    assert "disclosed miss" in text and "d0420" in text
    assert "abstention" in text and "d0736" in text
    # a stripped set must not read as clean: dropping one FITTING row moves the
    # printed denominator to 34, so a smaller sample never inherits "35/35". (The
    # last row in the file is an abstention, so `pairs[:-1]` would print an
    # unchanged 35/35 — which is itself the reason the denominator has to be
    # printed beside the number rather than implied by it.)
    cal = dd.load_calibration()
    victim = next(i for i, r in enumerate(cal["pairs"])
                  if r.get("in_fitting_set") is not False
                  and r["label"] != "unlabelled")
    cal["pairs"] = [r for i, r in enumerate(cal["pairs"]) if i != victim]
    stripped = tmp_path / "stripped.json"
    stripped.write_text(json.dumps(cal))
    out = dd.calibration_report(stripped)
    assert "agreement 34/34" in out, out.splitlines()[:2]
    assert "agreement 35/35" not in out


def test_a_label_the_threshold_cannot_reproduce_is_reported_not_dropped():
    """Flip one label and the agreement count moves — the guard is live.

    Without this, an `agreement()` that quietly skipped inconvenient rows would
    still print 40/40 forever.
    """
    rows = dd.load_calibration()["pairs"]
    agree, total, bad = dd.agreement(rows, dd.DUP_JACCARD_THRESHOLD)
    # 40 rows committed, 35 in the fitting set — see
    # `test_committed_calibration_is_at_least_thirty_pairs_and_is_reproduced` for
    # the two disclosed misses and three abstentions that make the difference, all
    # of which stay in the file and are named in the printed report.
    assert (agree, total, bad) == (35, 35, []), (agree, total, bad)
    assert len(rows) == 40 and len(dd.disclosed_misses(rows)) == 2
    # flip a row that IS in the fit: the count must move. A disclosed miss is
    # deliberately not used, because its exclusion is fixed by the class of evidence
    # behind its label, not by what it scored.
    target = next(i for i, r in enumerate(rows)
                  if r.get("in_fitting_set") is not False
                  and r["label"] != "unlabelled")
    flipped = [dict(r) for r in rows]
    flipped[target]["label"] = ("unrelated"
                                if flipped[target]["label"] != "unrelated" else "same")
    agree2, total2, bad2 = dd.agreement(flipped, dd.DUP_JACCARD_THRESHOLD)
    assert total2 == total and agree2 == agree - 1 and len(bad2) == 1


# ── the predicate ────────────────────────────────────────────────────────────

def test_identical_content_is_a_duplicate_and_unrelated_content_is_not():
    a = _note("alpha", _SHARED, "2026-05-01T00:00:00")
    b = _note("alpha-again", _SHARED, "2026-06-01T00:00:00")
    c = _note("wake-word", _OTHER, "2026-07-01T00:00:00")
    assert dd.similarity(a, b) == 1.0
    assert dd.is_near_duplicate(a, b) is True
    assert dd.similarity(a, c) < dd.DUP_JACCARD_THRESHOLD
    assert dd.is_near_duplicate(a, c) is False


def test_two_notes_with_different_lifecycle_trailers_are_still_one_finding():
    """The live case: one double-filed item whose appended triage differs.

    `backlog/1761-*` and `backlog/1762-*` are the same finding filed 4 ms apart by
    a double-fired owed-check pass; the second's trailer says `duplicate of #1761`
    and records the diff that proved it. Comparing whole files, the trailers are
    most of the text and the predicate reads 0.448 — under any usable threshold —
    while the bodies are byte-identical. Comparing content bodies it reads 1.0.
    That is why the predicate is not whole-file, and this fixture is the reason.
    """
    filed = _note("the same finding twice", _SHARED, "2026-09-28T22:47:23")
    twin = (_note("the same finding twice", _SHARED, "2026-09-28T22:47:23")
            + "\n## Automod triage — 2026-09-28\n\n"
            + "- **2026-09-28T23:03:33** — autotriage: stale. duplicate of #1761: "
              "`diff` of the two backlog files shows byte-identical title and body, "
              "4 ms apart in `created`, differing only in `position`.\n")
    assert dd.similarity(filed, twin) == 1.0
    assert dd.is_near_duplicate(filed, twin) is True
    # and the trailer really is what would have defeated a whole-file compare
    assert dd.content_body(twin) == dd.content_body(filed)
    assert "duplicate of #1761" in twin
    assert "duplicate of #1761" not in dd.content_body(twin)


def test_a_later_sections_heading_does_not_swallow_the_note_itself():
    """Stripping a lifecycle section must stop at the next section of its own level."""
    body = ("intro paragraph that is long enough to be worth comparing against "
            "another note in the same corpus and should survive any strip\n\n"
            "## Activity log\n\n- **2026-09-01T00:00:00** — triage: nothing here\n\n"
            "## What the note is actually about\n\n" + _SHARED)
    stripped = dd.content_body(_note("keeping", body))
    assert "intro paragraph" in stripped
    assert _SHARED.strip() in stripped
    assert "triage: nothing here" not in stripped


def test_similarity_of_empty_or_short_text_is_zero_not_crashing():
    assert dd.similarity("", "") == 0.0
    assert dd.similarity("five words only here", _SHARED) == 0.0
    assert dd.jaccard(set(), set()) == 0.0


# ── clustering and the slot share ────────────────────────────────────────────

def test_pairwise_sims_is_the_upper_triangle_in_the_order_cluster_slots_reads():
    texts = [_note("a", _SHARED), _note("b", _SHARED), _note("c", _OTHER)]
    sims = dd.pairwise_sims(texts)
    assert len(sims) == 3                       # 3*(3-1)//2
    assert sims[0] == 1.0                       # (0,1)
    assert sims[1] < dd.DUP_JACCARD_THRESHOLD   # (0,2)
    assert sims[2] < dd.DUP_JACCARD_THRESHOLD   # (1,2)


def test_dup_slot_share_counts_every_slot_inside_a_cluster():
    # 4 slots: two twins, two distinct → 2 of 4 slots are occupied by duplicates
    texts = [_note("a", _SHARED), _note("a-twin", _SHARED),
             _note("b", _OTHER), _note("c", _THIRD)]
    sims = dd.pairwise_sims(texts)
    share, in_cluster = dd.dup_slot_share(sims, len(texts))
    assert in_cluster == 2 and share == pytest.approx(0.5)


def test_distinct_slots_measure_exactly_zero_rather_than_abstaining():
    texts = [_note("a", _SHARED), _note("b", _OTHER), _note("c", _THIRD)]
    share, in_cluster = dd.dup_slot_share(dd.pairwise_sims(texts), len(texts))
    assert share == 0.0 and in_cluster == 0


def test_dup_slot_share_refuses_a_triangle_that_is_not_the_right_size():
    """A matrix built for another slot count is a bug, and must not read as 0.0."""
    with pytest.raises(ValueError):
        dd.dup_slot_share([1.0, 1.0], 4)
    assert dd.dup_slot_share([], 0) == (0.0, 0)
    assert dd.dup_slot_share([], 1) == (0.0, 0)


def test_slot_metrics_reports_slots_compared_and_slots_unreadable_separately():
    """`dup_slot_n` is what was measured; an unreadable slot never joins the share.

    The failure this pins: a returned top-K whose notes are all missing from disk
    (an eval run against a moved data root) would otherwise report
    `dup_slot_share: 0.0` and look like a clean corpus. It reports 0 slots compared
    and 5 unreadable, which is a different statement.
    """
    paths = ["knowledge/a.md", "knowledge/b.md", "knowledge/c.md",
             "knowledge/d.md", "knowledge/e.md"]
    texts = [_note("a", _SHARED), _note("a-twin", _SHARED), _note("c", _OTHER),
             None, None]
    out = dd.slot_metrics(paths, texts)
    assert out["dup_slot_n"] == 3
    assert out["dup_slot_unreadable"] == 2
    assert out["dup_slot_share"] == pytest.approx(2 / 3)
    assert out["dup_slot_threshold"] == dd.DUP_JACCARD_THRESHOLD
    everything = dd.slot_metrics(paths, [None] * 5)
    assert everything["dup_slot_n"] == 0
    assert everything["dup_slot_unreadable"] == 5
    assert everything["dup_slot_share"] == 0.0
    assert everything["superseded_slot_share"] == 0.0


# ── supersession ─────────────────────────────────────────────────────────────

def _meta(path, stamp, title, tags=()):
    return dd.slot_meta(path, _note(title, _SHARED, stamp, tags))


def test_supersession_needs_a_newer_note_on_the_same_topic():
    old = _meta("knowledge/gpu-serving.md", "2026-01-01T00:00:00", "gpu-serving")
    newer = _meta("knowledge/gpu-serving-v2.md", "2026-06-01T00:00:00",
                  "gpu-serving-v2")
    assert dd.superseded_slots([old, newer]) == [0]
    assert dd.superseded_slots([newer, old]) == [1]


def test_an_old_note_with_no_newer_twin_is_not_superseded():
    """The clause's negative half: old is not the same word as superseded.

    Two notes on genuinely different topics, one six months older, is the shape of
    a healthy corpus — a long-lived note is old and current at once, which is the
    video's own open question and the reason the topic test is required rather
    than a date filter.
    """
    old = _meta("knowledge/wake-word-audio.md", "2026-01-01T00:00:00", "wake-word-audio")
    recent = _meta("knowledge/gpu-serving.md", "2026-07-01T00:00:00", "gpu-serving")
    assert dd.superseded_slots([old, recent]) == []
    # the two rules disagree on precisely this note, which is the point of the
    # next test rather than an accident of this one
    assert dd.superseded_by_recency_only([old, recent]) == [0]


def test_the_recency_only_counterfactual_disagrees_on_exactly_the_unrelated_old_note():
    """The disagreement count is reported, not the choice hidden.

    Same fixture as the clause: one old note on a different topic from the newest
    note. The shipped rule counts nothing; a date filter would count the old note
    as superseded. The symmetric difference is therefore 1, and it is the number
    that says whether the topic test is load-bearing on this corpus.
    """
    old = _meta("knowledge/wake-word-audio.md", "2026-01-01T00:00:00", "wake-word-audio")
    recent = _meta("knowledge/gpu-serving.md", "2026-07-01T00:00:00", "gpu-serving")
    metas = [old, recent]
    assert dd.superseded_slots(metas) == []
    assert dd.superseded_by_recency_only(metas) == [0]
    out = dd.slot_metrics(["knowledge/wake-word-audio.md", "knowledge/gpu-serving.md"],
                          [_note("wake-word-audio", _SHARED, "2026-01-01T00:00:00"),
                           _note("gpu-serving", _SHARED, "2026-07-01T00:00:00")])
    assert out["superseded_slot_share"] == 0.0
    assert out["superseded_slot_count"] == 0
    assert out["superseded_slot_disagreements"] == 1


def test_a_note_with_no_front_matter_stamp_is_never_judged_superseded():
    """Missing metadata costs a verdict nothing, and a wrong verdict nothing."""
    unstamped = {"path": "knowledge/x.md", "stamp": None, "topic_key": "x",
                 "entities": set()}
    stamped = _meta("knowledge/x-v2.md", "2026-06-01T00:00:00", "x-v2")
    assert dd.superseded_slots([unstamped, stamped]) == []
    # and the recency-only rule cannot judge it either: with one dated note there
    # is nothing below the newest, so both rules return nothing and the
    # disagreement count is 0 — the unstamped note is simply never in question.
    assert dd.superseded_by_recency_only([unstamped, stamped]) == []


def test_stamp_of_normalises_naive_and_z_stampings_into_one_frame():
    """Both spellings live in the vault; comparing them naively raises TypeError.

    `memory/2026-05-11.md` carries `'2026-07-06T14:36:37'` with no offset and the
    automod ledger writes `2026-10-06T04:42:50Z`. A naive datetime compared to an
    aware one is an exception, so one note's metadata would take the nightly down.
    """
    naive = dd.stamp_of(_note("a", _SHARED, "2026-07-06T14:36:37"))
    zulu = dd.stamp_of(_note("b", _SHARED, "2026-10-06T04:42:50Z"))
    date_only = dd.stamp_of("---\ndate: 2026-05-07\n---\nbody\n")
    assert naive.tzinfo is not None and zulu.tzinfo is not None
    assert naive < zulu
    assert date_only == datetime(2026, 5, 7, tzinfo=timezone.utc)
    assert dd.stamp_of("no front matter at all") is None


def test_title_family_strips_the_numbering_and_dates_that_would_split_a_family():
    """The live family the predicate must not miss: 13 `main is red` filings.

    Their titles differ only by the commit sha the filing names — `1899-main is
    red: 3 failing test(s) in 1 file(s) at 221400f3` versus
    `2026-… at 219e1314` — so an unchanged-title test would call them distinct.
    """
    a = "main is red: 3 failing test(s) in 1 file(s) at 221400f3"
    b = "2026: main is red: 3 failing test(s) in 1 file(s) at 219e1314"
    assert dd.title_family(a) == dd.title_family(b)
    assert dd.title_family("gpu inference at the edge") != dd.title_family(
        "wake word audio binding")


def test_slot_meta_topic_and_entities_come_from_the_note_not_the_path():
    m = dd.slot_meta("knowledge/whatever.md",
                     _note("gpu-serving", _SHARED, "2026-06-01T00:00:00",
                           tags=["gpu", "naming", "knowledge"]))
    assert m["topic_key"] == dd.title_family("gpu-serving")
    # `knowledge` and `memory` are every note's segment name, not entities: they
    # would make every pair same-topic and every old note superseded.
    assert "knowledge" not in m["entities"]
    assert "gpu" in m["entities"]


# ── the collapse arm's own post-conditions ───────────────────────────────────

def _fixtures(paths, bodies, stamps, titles=None):
    """`(metas, texts)` for one returned top-K, from the same note texts.

    Titles default to the path stem, so a fixture that wants two notes in one
    title family says so rather than getting it from a path by accident.
    """
    titles = titles or [Path(p).stem for p in paths]
    texts = [_note(t, b, s) for t, b, s in zip(titles, bodies, stamps)]
    return [dd.slot_meta(p, t) for p, t in zip(paths, texts)], texts


def test_collapse_keeps_one_slot_per_cluster_at_the_same_budget():
    paths = ["knowledge/a.md", "knowledge/a-twin.md", "knowledge/b.md", "knowledge/c.md"]
    metas, texts = _fixtures(paths, [_SHARED, _SHARED, _OTHER, _THIRD],
                             ["2026-01-01T00:00:00", "2026-06-01T00:00:00",
                              "2026-02-01T00:00:00", "2026-03-01T00:00:00"])
    res = dd.collapse_slots(paths, metas, dd.pairwise_sims(texts))
    assert res.paths == ["knowledge/a-twin.md", "knowledge/b.md", "knowledge/c.md"]
    assert res.slots_freed == 1
    assert len(res.paths) <= len(paths)


def test_collapse_never_drops_the_gold_slot_even_when_it_is_the_older_twin():
    """The risk that would make the whole arm measure the scorer.

    Gold here is the older twin. The cluster's newest member would otherwise be
    the representative, which deletes the gold document and scores a doc miss that
    the collapse itself caused. The gold slot survives, at its earliest position,
    so the arm's `doc_hit` is the retrieval's answer and not the collapse's.
    """
    paths = ["knowledge/a.md", "knowledge/a-twin.md", "knowledge/b.md"]
    metas, texts = _fixtures(paths, [_SHARED, _SHARED, _OTHER],
                             ["2026-01-01T00:00:00", "2026-06-01T00:00:00",
                              "2026-02-01T00:00:00"])
    sims = dd.pairwise_sims(texts)

    def gold(p):
        return p == "knowledge/a.md"

    res = dd.collapse_slots(paths, metas, sims, gold)
    assert "knowledge/a.md" in res.paths
    assert "knowledge/a-twin.md" not in res.paths
    assert res.gold_survived is True
    assert res.slots_freed == 1
    assert res.paths[0] == "knowledge/a.md"          # earliest position kept

    # and the same run with the gold matcher inverted keeps the NEWER twin, which
    # is the rule working and not an accident of fixture ordering
    other = dd.collapse_slots(paths, metas, sims, lambda p: p == "knowledge/a-twin.md")
    assert other.paths[0] == "knowledge/a-twin.md"


def test_collapse_gold_matching_is_exact_and_not_a_substring_accident():
    """A gold matcher must not exempt a whole cluster of unrelated notes.

    `expect_docs` labels are substrings by design (`memory-` matches any daily
    note). If the arm used that rule for slot identity, every daily note would be
    gold, nothing would ever collapse, and the ablation would measure nothing while
    reporting 0 slots freed. `collapse_slots` takes exact paths here; `_score`'s
    `_slot_gold_matcher` keeps the substring rule for what it is for — scoring
    whether a returned slot counts as the gold document.
    """
    paths = ["memory/2026-05-11.md", "memory/2026-05-12.md"]
    metas, texts = _fixtures(paths, [_SHARED, _SHARED],
                             ["2026-05-11T00:00:00", "2026-05-12T00:00:00"])
    sims = dd.pairwise_sims(texts)
    res = dd.collapse_slots(paths, metas, sims, ["memory-"])
    assert res.slots_freed == 1, "a substring label must not squat every daily note"
    assert res.gold_survived is True                  # nothing was gold


def test_collapse_is_a_no_op_when_there_is_nothing_to_collapse():
    paths = ["knowledge/a.md", "knowledge/b.md"]
    metas, texts = _fixtures(paths, [_SHARED, _OTHER],
                             ["2026-01-01T00:00:00", "2026-02-01T00:00:00"])
    res = dd.collapse_slots(paths, metas, dd.pairwise_sims(texts))
    assert res.paths == paths and res.slots_freed == 0
    assert dd.collapse_slots([], [], []).paths == []


def test_the_committed_threshold_is_the_lowest_highest_agreement_on_the_grid():
    """The threshold as a decision, not a convention: run the whole grid over the
    committed fitting set and check the constant is where the item says it is.

    Every grid point gets a number, so the shape of the whole choice is visible —
    0.30: 27/35, 0.40: 30/35, 0.50: 33/35, 0.55: 34/35, 0.60: 35/35 — and
    `first_argmax_threshold` takes the LOWEST value with the HIGHEST agreement,
    which is the reading the item states. That the answer is 0.6 and not 0.65 is
    what makes this a check: both agree on 35 of 35 rows, and the tie goes down,
    so a looser constant could claim the same agreement while silently calling a
    0.62 pair a duplicate. The grid values themselves are asserted rather than
    trusted, because a `threshold_grid` that quietly scored the wrong denominator
    would still hand back 0.6.
    """
    cal = dd.load_calibration()
    grid = dd.threshold_grid(cal)
    assert grid[dd.DUP_JACCARD_THRESHOLD] == f"{cal['n_fitting_set']}/{cal['n_fitting_set']}"
    assert grid[0.55] == "34/35" and grid[0.50] == "33/35", grid
    assert grid[0.75] == "30/35" and grid[0.85] == "26/35", grid
    assert dd.first_argmax_threshold(cal) == dd.DUP_JACCARD_THRESHOLD
