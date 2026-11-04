"""The complement probe's three arms, its edge-disjoint gold and its artifact (#3642 clause 5).

Clause 5: one artifact reporting ``complement_hit_rate`` for warp-off, warp-on and
control-on over gold pairs whose defining edges are disjoint from the edges the warp
was fitted on, printing both edge-id sets and their (empty) intersection, alongside
the current embedder's measured mean pairwise cosine and the vector dimension read.

The fixture is built so the arms can be told apart rather than merely enumerated:
the fit-side neighbour and the gold complement share an off-axis direction the seed
does not contain, so the neighbour-evidence residual points at the gold target while
the random control direction has no reason to. That is what makes ``warp-on``
beating ``control-on`` a checkable claim instead of a hope — and if the fixture's
fixed control direction happens to land near the target, the assertion below says so
instead of the probe quietly reporting a win.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieval" / "complement_probe.py"
sys.path.insert(0, str(ROOT / "tests"))

from _complement_fixture import write_kg, write_qmd_index  # noqa: E402

DIM = 8
# The seed sits at 0.196 of cosine off axis e1; the fit neighbour and the gold
# complement both carry a large e1 share, so the complement evidence has somewhere
# to point that the seed does not already occupy.
SEED = [1.0, 0.2, 0, 0, 0, 0, 0, 0]
GOLD = [0.35, 0.9368, 0, 0, 0, 0, 0, 0]
FIT = [0.30, 0.954, 0, 0, 0, 0, 0, 0]
# Four distractors ranked between them off-axis in e2: nearer to the seed than the
# gold target is, so an unwarped seed ranks the target below the cut.
DIST = [[0.6, 0, 0.8, 0, 0, 0, 0, 0] for _ in range(4)]
# Two SUBSTITUTES of the seed, one per route into the strong set, and both in the
# (e0,e2) plane so they carry no share of the off-axis direction (e1) the complement
# evidence points along. SUB_HIGH reaches cosine 0.7355 from the seed and is joined
# by NO edge, so it can only enter the strong set by the high-cosine rule; SUB_EDGE
# reaches only 0.343, so it can only enter by its `competes_with` edge. Together they
# make the strong set two rows deep — the smallest set with a cosine GAP in it, which
# is what the gap-preserving limit needs (`strong_u.shape[0] < 2` returns the cap
# untouched), so this fixture exercises the conceptor AND the disturbance limit.
SUB_HIGH = [0.75, 0, 0.6614, 0, 0, 0, 0, 0]
SUB_EDGE = [0.35, 0, 0.9368, 0, 0, 0, 0, 0]
NOTES = {
    "knowledge/SeedEnt.md": [SEED],
    "knowledge/GoldEnt.md": [GOLD],
    "knowledge/FitEnt.md": [FIT],
    "knowledge/SubHigh.md": [SUB_HIGH],
    "knowledge/SubEdge.md": [SUB_EDGE],
    "knowledge/d1.md": [DIST[0]],
    "knowledge/d2.md": [DIST[1]],
    "knowledge/d3.md": [DIST[2]],
    "knowledge/d4.md": [DIST[3]],
}
# Ordered so the parity split puts the FIT evidence in the fit half (id 1) and the
# scored gold pair in the gold half (id 2) — the split is by id, so this ordering IS
# the experiment's design and a reader can check it from these six lines.
# Appended LAST so its id is 3: `ordered_edges` filters to the complement types
# before the parity split, so a substitute edge must not be able to move the fit/gold
# ids of the pairs already above it.
EDGES = [
    ("SeedEnt", "FitEnt", "uses", "notes/absent.md"),
    ("SeedEnt", "GoldEnt", "uses", "notes/absent.md"),
    ("SeedEnt", "SubEdge", "competes_with", "notes/absent.md"),
]


def _run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    index = write_qmd_index(tmp_path / "index.sqlite", DIM, NOTES)
    kg = write_kg(tmp_path / "kg.sqlite", EDGES)
    dest = tmp_path / "_pipeline" / "vault-derived" / "kg.sqlite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(kg.read_bytes())
    child = dict(os.environ)
    child["LLOYD_DATA"] = str(tmp_path)      # app.data_root's first candidate

    child["QMD_INDEX"] = str(index)          # the env route, as a real run would use
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(tmp_path),
         "--out", str(tmp_path / "probe.json"), "--k", "3", *extra],
        capture_output=True, text=True, cwd=str(ROOT), env=child, timeout=300)


def test_one_artifact_reports_all_three_arms_over_edge_disjoint_gold(tmp_path):
    """Clause 5: three rates, one artifact, gold edges disjoint from fit edges."""
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((tmp_path / "probe.json").read_text())
    assert report["kind"] == "complement_probe" and report["item"] == 3642
    rates = report["complement_hit_rate"]
    assert set(rates) == {"off", "warp", "control"}, rates
    for arm, value in rates.items():
        assert isinstance(value, float) and 0.0 <= value <= 1.0, (arm, value)
    assert report["gold_pairs_scored"] == 1, "the fixture has exactly one gold pair"
    split = report["edge_split"]
    assert split["fit_edge_ids"] == [1] and split["gold_edge_ids"] == [2], split
    assert split["intersection"] == [], "the gold edge may not be in the fit set"
    scored = report["per_pair"][0]
    assert scored["edge_id"] == 2 and scored["edge_id"] not in split["fit_edge_ids"]
    assert scored["fit_neighbours"] >= 1, "the seed must carry fit-side evidence"
    # The store the rates were measured over has to be identifiable after the fact:
    # clause 5 names the embedder, the dimension AND the fingerprint, because a rate
    # from an index that has since been re-embedded is not the same measurement.
    assert report["embedder"].startswith("fixture:")
    assert report["vector_dim"] == DIM
    assert report["store_fingerprint"], "clause 5: the artifact must name the store"
    assert f"store fingerprint:   {report['store_fingerprint']}" in proc.stdout, proc.stdout
    for arm in ("off", "warp", "control"):
        assert scored["arms"][arm]["hit"] in (True, False)
    # the arms actually did something: an injection length on the unit seed
    assert scored["arms"]["off"]["injection"] == 0.0
    assert scored["arms"]["warp"]["injection"] > 0.0
    assert scored["arms"]["control"]["injection"] > 0.0


def test_the_probe_prints_both_edge_id_sets_their_intersection_and_the_geometry(tmp_path):
    """Clause 5, printed half: the disjointness and the embedder figures are on stdout."""
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "fit-side edge ids  (1):  [1]" in out, out
    assert "gold-side edge ids (1):  [2]" in out, out
    assert "intersection of the two edge-id sets: (empty)" in out, out
    assert f"vector dim:          {DIM}" in out, out
    assert "embedder:            fixture:Test-Embedding-0.6B.gguf" in out, out
    assert "mean pairwise cosine" in out, out
    for arm in ("off", "warp", "control"):
        assert f"  {arm:8s} complement_hit_rate=" in out, out
    # one arm zero and one arm not-zero is the printed shape that makes the three
    # arms a comparison rather than three copies of one number
    assert "  off      complement_hit_rate=0.0" in out, out
    assert "  warp     complement_hit_rate=1.0" in out, out


def test_the_neighbour_evidence_outranks_the_random_direction_on_this_fixture(tmp_path):
    """The comparison that decides anything, checked where it can be checked.

    The gold target starts below the cut (four distractors sit nearer the seed), and
    the only off-axis evidence the warp is given points at it, so warp-on must rank
    it no worse than warp-off and better than the random control at the same cap.
    If a future change makes the control arm match the warp arm here, that is the
    null result the item says to report — and this test is where it surfaces.
    """
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((tmp_path / "probe.json").read_text())
    rates = report["complement_hit_rate"]
    assert rates["warp"] > rates["off"], rates
    assert rates["warp"] > rates["control"], rates
    assert report["control_verdict"] == "warp-ahead", report["control_verdict"]
    ranks = report["per_pair"][0]["arms"]
    assert ranks["warp"]["rank"] < ranks["off"]["rank"], ranks
    assert ranks["warp"]["hit"] is True and ranks["off"]["hit"] is False, ranks


SHARED_NOTES = dict(NOTES, **{
    # The one note both heads of the shared pair resolve to: two chunks, so the
    # entity vector is a mean, as on live data.
    "knowledge/SharedEvid.md": [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
    "knowledge/AlphaOnly.md": [[1, 0, 0, 0, 0, 0, 0, 0]],
    "knowledge/BetaOnly.md": [[0, 0, 0, 0, 1, 0, 0, 0]],
})
# EDGES above stays first, so ids 1 and 2 keep their fit/gold roles. Id 3 is another
# fit-side pair; id 4 is the shared-note pair, which lands on the GOLD side. AlphaHead
# and BetaHead have no note named for them, so both fall to the source-document route
# and resolve onto knowledge/SharedEvid.md — the note their own edge was extracted
# from. Their private evidence rides on `related_to`, which makes it a source document
# without becoming a pair the sweep could count. Their entity cosine is 0.2222, inside
# [0.20, 0.55): the pair clears the cosine window on chunks both endpoints own.
SHARED_EDGES = EDGES + [
    ("CompD", "CompB", "uses", "notes/absent-d.md"),
    ("AlphaHead", "BetaHead", "uses", "knowledge/SharedEvid.md"),
    ("AlphaHead", "SeedA", "related_to", "knowledge/AlphaOnly.md"),
    ("BetaHead", "CompB", "related_to", "knowledge/BetaOnly.md"),
]


def _run_shared(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """The same run as :func:`_run`, over the shared-note fixture, with LLOYD_DATA empty.

    ``LLOYD_PRODUCTION_DATA=""`` is not decoration: without it the real data root can
    answer for a fixture and the test measures production. The harvest probe's tests set
    it for the same reason.
    """
    index = write_qmd_index(tmp_path / "index.sqlite", DIM, SHARED_NOTES)
    kg = write_kg(tmp_path / "kg.sqlite", SHARED_EDGES)
    root = tmp_path / "root"
    dest = root / "_pipeline" / "vault-derived" / "kg.sqlite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(kg.read_bytes())
    child = dict(os.environ)
    child.pop("LLOYD_DATA", None)
    child["LLOYD_PRODUCTION_DATA"] = ""
    child["QMD_INDEX"] = str(index)
    out = root / "eval" / "baselines" / "probe.json"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--data-root", str(root), "--out", str(out),
         "--k", "3", *extra],
        capture_output=True, text=True, cwd=str(ROOT), env=child, timeout=300)


def test_a_shared_note_gold_pair_is_stripped_and_its_count_is_published(tmp_path):
    """Clause 5, the strip: a gold pair that cannot fail never reaches the scorer.

    Scoring one reports a rate inflated by an identity — every arm ranks the target
    first because the target's chunks ARE the seed's chunks — and the item's live
    measurement is that this shape accounts for a large slice of the complement pairs.
    So the run must still score the clean gold pair, must publish how many pairs it
    removed, and must keep a stripped pair out of the per-pair list it reports.
    """
    proc = _run_shared(tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = tmp_path / "root" / "eval" / "baselines" / "probe.json"
    art = json.loads(out.read_text())
    assert art["gold_pairs_endpoints_shared_note_excluded"] == 1, art
    assert "endpoints resolve to the same note" in proc.stdout, proc.stdout
    assert art["edge_split"]["gold_edge_ids"] == [2], art["edge_split"]
    assert art["gold_pairs_scored"] == 1, art
    assert [pp["edge_id"] for pp in art["per_pair"]] == [2], art["per_pair"]


def test_a_gold_pair_reaching_the_scorer_stops_the_run_from_reporting_a_rate(tmp_path):
    """Clause 5, the refusal: a shared note in the SCORED gold set means no artifact.

    ``--show-same-note-gold`` deliberately keeps the pairs an ordinary run strips, which
    is the only way to drive the guard: in a normal run the strip happens upstream, so
    the refusing branch is otherwise unreachable, and an unexercised guard is a guard
    nobody has seen fire. What has to hold is that the refusal is TOTAL — no rate on
    stdout, no artifact on disk, non-zero exit — because a half-suppressed run is exactly
    how a self-measuring number gets into an artifact people quote.
    """
    proc = _run_shared(tmp_path, "--show-same-note-gold")
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "REFUSED" in proc.stderr, proc.stderr
    assert "AlphaHead" in proc.stderr and "BetaHead" in proc.stderr, proc.stderr
    assert not (tmp_path / "root" / "eval" / "baselines" / "probe.json").exists(), \
        "a run that refused still wrote an artifact"
    assert "complement_hit_rate" not in proc.stdout, proc.stdout


def test_the_strong_set_is_the_seeds_substitutes_and_not_its_own_chunks(tmp_path):
    """Clause 5's substance: the arm reported as the Stage-1 warp gets substitute evidence.

    The first shipped probe built the strong set from "the seed note's other chunks",
    which is not a substitute set (two chunks of one note measured 0.877 apart on the
    live store — near-duplicates of the seed) and was almost always EMPTY, because
    9,022 of 9,411 live notes are single-chunk. An empty strong set leaves
    ``_conceptor_removal`` a stack of one and short-circuits the gap-preserving limit
    at ``strong_u.shape[0] < 2``, so the number printed under ``warp`` came from a
    neighbour mean with neither of Stage 1's two mechanisms running.

    Each route is pinned separately, by which note appears: ``SubHigh`` has no edge at
    all and is in the strong set only because its cosine to the seed (0.7355) clears
    the 0.70 substitute cut, while ``SubEdge`` reaches only 0.343 and is in it only
    because of its ``competes_with`` edge. And the seed's OWN note is absent, which is
    the half that was wrong.
    """
    proc = _run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    report = json.loads((tmp_path / "probe.json").read_text())
    summary = report["strong_set"]
    assert summary["pairs_scored"] == 1, summary
    assert summary["pairs_with_empty_strong_set"] == 0, summary
    assert summary["min"] >= 2, \
        f"one substitute row means no cosine gap exists, so the gap limit stays inert: {summary}"
    scored = report["per_pair"][0]
    paths = scored["strong_paths"]
    assert scored["strong_neighbours"] == summary["min"], (scored, summary)
    assert any(p.endswith("SubHigh.md") for p in paths), \
        f"the high-cosine route is missing, so the conceptor has no competitors: {paths}"
    assert any(p.endswith("SubEdge.md") for p in paths), \
        f"the typed substitute route is missing: {paths}"
    assert not any(p.endswith("SeedEnt.md") for p in paths), \
        f"the seed's own note is not its substitute: {paths}"
