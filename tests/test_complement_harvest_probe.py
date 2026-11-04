"""The harvest probe's counts, its data-root resolution and its artifact (#3642 clauses 1, 2).

Clause 1: one command, no service restart, printing the active-edge type
histogram, the usable complement-pair count over pairs whose two endpoints resolve
to chunks present in the qmd index AND to disjoint note-path sets, the number of
seeds with at least one such pair, and writing a JSON artifact into the
eval-baselines directory of the resolved data root. Disjointness is part of the
clause, not a refinement of it: a pair whose endpoints are the same note asks which
chunk is nearest to itself and always finds it, so counting one inflates the harvest
with pairs no arm can fail. The probe therefore prints how many pairs it excluded
that way instead of dropping them silently — the size of that exclusion is itself
the entity-linking finding this item owes, and a test that asserted the live count
would be a number nobody re-derives (the #3719/#3764 class), so the numbers live in
the artifact and only the MECHANISM is pinned here.

Clause 2: run from a tree with no data directory of its own, it still reports live
counts and prints the root it read — because it resolves through
``app.data_root.production_data_root()`` / ``kg_store_for_root()`` and not through
anything repo-relative. That is the failure that cost #577 its clause 6: an automod
worktree has no ``_pipeline/vault-derived/kg.sqlite``, so a repo-relative read
there returns zeros that look like an empty graph instead of an absent one.

Both are exercised over a fixture pair — a KG store built by ``app.kg_store`` and
a scored index whose tables match the ones ``qmd_vectors`` reads — because the
live stores are read-only, >1 MB apart from the clause's point, and a test must not
depend on what the daemon happens to have indexed today.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "retrieval" / "complement_harvest_probe.py"
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402

from _complement_fixture import write_kg, write_qmd_index  # noqa: E402
from app.data_root import kg_store_for_root, production_data_root  # noqa: E402

DIM = 8
A1 = [1, 0.1, 0, 0, 0, 0, 0, 0]
A2 = [1, 0, 0.1, 0, 0, 0, 0, 0]
B = [0.3, 0, 0, 0.95, 0, 0, 0, 0]
C = [0.98, 0, 0, 0, 0.2, 0, 0, 0]
D = [0, 0, 0, 0.4, 0.92, 0, 0, 0]
E = [0.8, 0, 0, 0.4, 0.3, 0, 0, 0]
NOTES = {
    # SeedA carries two chunks, so its entity vector is a mean, as on live data.
    # The note stems match the entity names, which is the per-endpoint join
    # `note_titles` implements; the edges' source_doc values below are deliberately
    # notes that are NOT indexed, so the title route is what resolves here.
    "knowledge/SeedA.md": [A1, A2],
    "knowledge/CompB.md": [B],
    "knowledge/SubC.md": [C],
    "knowledge/CompD.md": [D],
}
# Entity-vector cosines the fixture is built to produce: A·B 0.299, D·B 0.380 and
# B·C 0.486 all inside [0.20, 0.55); A·D 0.00 below the floor; A·C 0.987 is the one
# substitute pair (the competes_with edge, cosine >= 0.70).
EDGES = [
    ("SeedA", "CompB", "uses", "notes/absent-a.md"),
    ("CompD", "CompB", "uses", "notes/absent-d.md"),
    ("CompB", "SubC", "part_of", "notes/absent-b.md"),
    ("SeedA", "CompD", "depends_on", "notes/absent-a.md"),
    ("SeedA", "SubC", "competes_with", "notes/absent-a.md"),
    ("Ghosts", "Nowhere", "uses", "notes/absent-e.md"),
]
# The expected figures are written as literals at each assertion, not as module
# constants: a constant a fixture edit can be nudged alongside is not an expectation
# any more. What they are: the three in-window complement pairs (edges 1, 2 and 3),
# and the four entities that carry at least one of them (SeedA, CompB, SubC, CompD).


def _stores(tmp_path: Path) -> tuple[Path, Path]:
    """A scored index and a KG store, the KG at the path `kg_store_for_root` expects."""
    index = write_qmd_index(tmp_path / "index.sqlite", DIM, NOTES)
    kg = write_kg(tmp_path / "kg.sqlite", EDGES)
    return index, kg


def _install_kg(root: Path, kg: Path) -> Path:
    """Move the fixture KG to <root>/_pipeline/vault-derived/kg.sqlite."""
    dest = root / "_pipeline" / "vault-derived" / "kg.sqlite"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(kg.read_bytes())
    return dest


def _run(args, *, cwd=None, env=None) -> subprocess.CompletedProcess:
    child = dict(os.environ)
    child.pop("LLOYD_DATA", None)       # the suite's scratch root must not leak
    child["LLOYD_PRODUCTION_DATA"] = ""  # never let the real root answer for a fixture
    child.update(env or {})
    # A `None` in `env` means "this variable must not exist in the child", which is
    # the only way a node can express clause 2's condition — nothing named. Passing
    # None through to subprocess.run is a TypeError, not an unset.
    for _k, _v in list((env or {}).items()):
        if _v is None:
            child.pop(_k, None)
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
        cwd=str(cwd or ROOT), env=child, timeout=300)


def test_one_command_prints_the_histogram_pair_count_and_seed_count(tmp_path):
    """Clause 1: the four numbers, from one command, with no service restart."""
    index, kg = _stores(tmp_path)
    _install_kg(tmp_path, kg)
    proc = _run(["--index", str(index), "--data-root", str(tmp_path),
                 "--out", str(tmp_path / "art.json")])
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    assert "active edge type histogram (6 active edges)" in out, out
    for line in ("  uses                   3 <- complement",
                 "  depends_on             1 <- complement",
                 "  part_of                1 <- complement",
                 "  competes_with          1 <- substitute"):
        assert line in out, f"missing histogram line {line!r} in:\n{out}"
    assert "USABLE complement pairs (cosine in [0.2, 0.55)): 3" in out, out
    assert "SEEDS with >= 1 usable complement pair:                    4" in out, out
    # the edge whose endpoints have no indexed note must not have been counted
    assert ("complement-class active edges with both endpoints vectorised: 4"
            in out), out
    assert "substitute-class pairs (typed, cosine >= 0.7):  1" in out, out
    # the item's stop rule is a printed outcome, not an implied one
    assert "STOP RULE" in out and "fewer than 50" in out, out


def test_the_probe_writes_its_artifact_into_the_resolved_roots_eval_baselines(tmp_path):
    """Clause 1, second half: the artifact lives under the root it resolved."""
    index, kg = _stores(tmp_path)
    _install_kg(tmp_path, kg)
    out = tmp_path / "art.json"
    proc = _run(["--index", str(index), "--data-root", str(tmp_path), "--out", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert out.exists(), out
    report = json.loads(out.read_text())
    assert report["kind"] == "complement_harvest_probe" and report["item"] == 3642
    assert report["data_root"] == str(tmp_path)
    assert report["active_edge_histogram"] == {
        "uses": 3, "depends_on": 1, "part_of": 1, "competes_with": 1}
    assert report["usable_complement_pairs"] == 3
    assert report["seeds_with_usable_complement_pair"] == 4
    assert report["usable_complement_pairs_below_stop_rule"] is True
    assert report["entities_with_source_doc"] == 6
    assert report["entity_resolution"]["entities_resolved"] == 4, \
        "only the four named notes resolve; Ghosts and Nowhere have no indexed note"
    assert report["entity_resolution"]["entities_vectorised"] == 4
    assert report["complement_types"] == ["uses", "depends_on", "part_of"]
    assert report["substitute_pairs"] == 1, "A·C cosine 0.987 is the only substitute pair"
    assert report["vector_dim"] == DIM and report["embedder"].startswith("fixture:")
    assert report["cosine_diagnostics"]["mean_pairwise_cosine"] is not None
    assert report["store_fingerprint"].startswith("content_vectors=5")
    # the pair list is the same count the summary carries, and edge ids are recorded
    assert len(report["pairs_sample"]) == 3
    assert {p["edge_id"] for p in report["pairs_sample"]} == {1, 2, 3}


def test_the_default_artifact_path_is_under_the_resolved_root(tmp_path):
    """No --out: the artifact still lands in <resolved root>/eval/baselines."""
    index, kg = _stores(tmp_path)
    _install_kg(tmp_path, kg)
    proc = _run(["--index", str(index), "--data-root", str(tmp_path)])
    assert proc.returncode == 0, proc.stderr
    written = list((tmp_path / "eval" / "baselines").glob("complement-harvest-*.json"))
    assert len(written) == 1, proc.stdout
    assert "artifact:" in proc.stdout and str(written[0]) in proc.stdout


def test_from_an_empty_tree_with_the_environment_named_it_uses_that_root(tmp_path):
    """``$LLOYD_DATA`` is honoured, and the artifact follows it.

    This is the PRECEDENCE case, not clause 2's branch: the caller named a root, so
    the probe reads it and nothing else. Clause 2's branch — nothing named at all —
    is ``test_with_no_environment_and_no_flag_it_still_reports_live_counts``, and the
    two are separate nodes precisely because the first round's review could not tell
    them apart, and neither can a reader who is shown only one of them.
    """
    index, kg = _stores(tmp_path)
    root = tmp_path / "root"
    _install_kg(root, kg)
    empty_tree = tmp_path / "tree"
    empty_tree.mkdir()
    proc = _run(["--index", str(index)], cwd=empty_tree, env={"LLOYD_DATA": str(root)})
    assert proc.returncode == 0, proc.stderr
    assert f"data root read:      {root}  (via app.data_root.resolve_data_root_for_tree" \
        in proc.stdout, proc.stdout
    assert f"kg store:            {root / '_pipeline' / 'vault-derived' / 'kg.sqlite'}" \
        in proc.stdout, proc.stdout
    assert "USABLE complement pairs (cosine in [0.2, 0.55)): 3" in proc.stdout
    assert "SEEDS with >= 1 usable complement pair:                    4" in proc.stdout
    assert list((root / "eval" / "baselines").glob("complement-harvest-*.json")), \
        "the artifact followed the resolved root, not the tree it ran from"


def test_a_pair_whose_endpoints_resolve_to_one_note_is_excluded_and_counted(tmp_path):
    """Clause 1's "disjoint note-path sets" half, plus the count that keeps it auditable.

    AlphaHead and BetaHead are two entities that co-resolve onto one note: neither has
    a note named for it, so both fall to the source-document route, and the note their
    edge was extracted from — `knowledge/SharedEvid.md`, which IS indexed — lands in
    both resolved sets. Their entity vectors are means that share those two chunks and
    differ by one private note each, which puts their cosine at 0.2222, INSIDE
    [0.20, 0.55). That is what makes the pair dangerous rather than merely odd: it
    clears the cosine window on the strength of evidence both endpoints own, so a probe
    over it is asking which chunk is nearest to this chunk, and every arm answers 1.0.

    The pair therefore has to stay out of `usable_complement_pairs`, and the exclusion
    has to be PRINTED — how many pairs are this shape is itself an entity-linking
    finding #3642 owes, and a silently dropped pair reads as a thin graph instead. The
    three real pairs and the seed count are asserted unchanged in the same breath: an
    exclusion that also ate genuine pairs would keep the excluded figure honest and the
    harvest wrong, which is the failure on the other side of the same knob.
    """
    notes = dict(NOTES, **{
        # Two chunks, so a mean — and the one note both endpoints end up resolving to.
        "knowledge/SharedEvid.md": [[0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
        "knowledge/AlphaOnly.md": [[1, 0, 0, 0, 0, 0, 0, 0]],
        "knowledge/BetaOnly.md": [[0, 0, 0, 0, 1, 0, 0, 0]],
    })
    edges = list(EDGES) + [
        ("AlphaHead", "BetaHead", "uses", "knowledge/SharedEvid.md"),
        # The private evidence arrives on a NON-complement type, so it contributes a
        # source document without becoming a pair the sweep could count: what is under
        # test is the note filter alone, with the cosine window unchanged around it.
        ("AlphaHead", "SeedA", "related_to", "knowledge/AlphaOnly.md"),
        ("BetaHead", "CompB", "related_to", "knowledge/BetaOnly.md"),
    ]
    index = write_qmd_index(tmp_path / "i2.sqlite", DIM, notes)
    kg = write_kg(tmp_path / "kg2.sqlite", edges)
    root = tmp_path / "r3"
    (root / "eval" / "baselines").mkdir(parents=True)
    _install_kg(root, kg)
    out = root / "eval" / "baselines" / "shared.json"
    proc = _run(["--index", str(index), "--data-root", str(root), "--out", str(out)])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    art = json.loads(out.read_text())

    assert art["complement_pairs_endpoints_shared_note_excluded"] == 1, art
    assert "endpoints resolve to the same note" in proc.stdout, proc.stdout
    # `pairs_sample` is the artifact's list of usable pairs (capped at 50; with
    # USABLE_PAIRS == 3 here it is the whole set, which is what makes membership
    # decidable from the artifact alone).
    assert art["usable_complement_pairs"] == len(art["pairs_sample"]), art
    usable = {(p["source"], p["target"]) for p in art["pairs_sample"]}
    assert ("AlphaHead", "BetaHead") not in usable, \
        "a shared-note pair was counted usable, so the harvest can be inflated by pairs " \
        "no arm is able to fail"
    assert {("SeedA", "CompB"), ("CompD", "CompB"), ("CompB", "SubC")} <= usable, \
        "excluding the shared-note pair cost a real pair as well"
    assert art["usable_complement_pairs"] == 3, art["usable_complement_pairs"]
    assert art["seeds_with_usable_complement_pair"] == 4, \
        art["seeds_with_usable_complement_pair"]


def test_a_tree_with_no_data_directory_at_all_says_so_instead_of_reporting_zero(tmp_path):
    """An absent store is a message, not a histogram of zeros.

    This is the specific hole #577 fell into: in a worktree with no
    ``_pipeline/vault-derived/``, a repo-relative or created-empty read returns 0
    edges, and 0 printed under a heading that says "active edge type histogram"
    reads as an empty graph instead of a missing one.
    """
    index, _kg = _stores(tmp_path)
    empty_root = tmp_path / "nodata"
    (empty_root / "_pipeline" / "vault-derived").mkdir(parents=True)
    empty_tree = tmp_path / "tree2"
    empty_tree.mkdir()
    proc = _run(["--index", str(index), "--data-root", str(empty_root)],
                cwd=empty_tree)
    assert proc.returncode == 2, proc.stdout
    assert "no KG store at" in proc.stderr, proc.stderr
    assert "histogram" not in proc.stdout.lower(), proc.stdout


def test_the_probe_never_reads_a_repo_relative_kg_path():
    """Clause 2's static half: the script resolves, it does not concatenate.

    ``.lloyd-data`` beside the script and any path built from the repo root are
    both ways to get a store that is not the live one, so neither may appear in the
    source; the resolution must go through the two accessors named in the clause.
    """
    src = SCRIPT.read_text()
    assert "production_data_root()" in src and "kg_store_for_root(" in src
    for banned in ('".lloyd-data"', '"_pipeline/vault-derived"',
                   "'_pipeline/vault-derived'", 'REPO / "eval"'):
        assert banned not in src, f"{banned} is a repo-relative data path: clause 2's bug"
    for banned in ("rm -rf", "TRUNCATE", "delete from edges"):
        assert banned not in src, "the harvest probe must not write the live KG"


def test_with_no_environment_and_no_flag_it_still_reports_live_counts(tmp_path):
    """Clause 2's branch: NO `--data-root`, NO `LLOYD_DATA`, a tree with no store.

    The environment the child gets has `LLOYD_DATA` removed rather than pointed at a
    fixture, because the point of the clause is what happens when NOTHING is named:
    `app.data_root` then answers with a root derived from the tree, an automod
    worktree's own root is `<worktree>/.lloyd-data`, and that directory does not
    exist — `resolve_data_root_for_tree()` returns it without raising, so an
    exception-only fallback never fires and the probe would report a histogram of
    zeros over a store that is simply not there. That is the #577 failure shape, and
    it is a passing test only if the run reports the LIVE counts instead.

    `--dry-run` is not decoration. Without it this node writes its artifact into the
    production data root, which is where the five stray
    `complement-harvest-20261104-09*.json` files the groundskeeper is now asked to
    clear came from: a test that resolves to the live root must read it but must not
    litter it.
    """
    from app.data_root import kg_store_for_root, production_data_root

    live_root = production_data_root()
    live_kg = kg_store_for_root(live_root)
    if not live_kg.is_file():
        pytest.skip(f"no live KG store at {live_kg}; clause 2 is about finding it")
    empty_tree = tmp_path / "tree"
    empty_tree.mkdir()
    proc = _run(["--dry-run"], cwd=empty_tree, env={"LLOYD_DATA": None})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BLOCKER: no KG store" not in proc.stderr, proc.stderr
    assert f"data root read:      {live_root}  (via app.data_root." in proc.stdout, \
        proc.stdout
    assert f"kg store:            {live_kg}" in proc.stdout, proc.stdout
    # Live counts, not the zeros of an absent store: the figure is whatever the live
    # graph holds today, so only "it is a real number and the header printed" is
    # assertable here — the counts themselves belong in the artifact (clause 1).
    shown = re.search(r"active edge type histogram \((\d+) active edges\)", proc.stdout)
    assert shown and int(shown.group(1)) > 0, proc.stdout
    # The band's own closing ")" sits inside the parenthetical, so the pattern has to
    # run to the end of the line rather than to the first ")".
    usable = re.search(r"USABLE complement pairs \(cosine in [^\n]*\): (\d+)", proc.stdout)
    assert usable and int(usable.group(1)) > 0, proc.stdout
    assert f"{live_root / 'eval' / 'baselines'}" in proc.stdout, \
        "the default artifact path did not follow the resolved root"
    assert "--dry-run" in proc.stdout


def test_a_tree_root_naming_no_store_falls_back_and_a_named_root_never_does(tmp_path,
                                                                            monkeypatch):
    """The fallback's two shapes, in-process, and the half that must NOT fall back.

    `resolve_roots` has to survive both ways a tree can lack a store — the resolver
    raising, and the resolver returning a root that simply has no KG in it, which is
    what an automod worktree does — and it must NOT rescue a root the caller named:
    silently reading a different store than the one `--data-root` gave would make
    every sandboxed number a possible live number.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("chp_fallback", SCRIPT)
    chp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chp)

    class Boom(RuntimeError):
        pass

    def boom():
        raise Boom("no resolvable root")

    original = chp.resolve_data_root_for_tree
    # conftest pins LLOYD_DATA to a scratch root for the whole suite; the fallback is
    # deliberately gated on the environment naming NOTHING, so this node has to take
    # that pin off to be the branch it claims to cover. `monkeypatch` rather than an
    # `os.environ[...]` assignment is not stylistic: `scripts/automod/code_rules.toml`
    # `exports-lloyd-data-to-process-environ` forbids a test writing that variable into
    # the process (every child inherits it), and its two allowlist entries are the
    # suite-wide conftest pin and one save/restore site. pytest reverts this node's
    # change on its own, which is the same guarantee with no leaked child env.
    monkeypatch.delenv("LLOYD_DATA", raising=False)
    try:
        # Shape 1: the resolver itself fails.
        chp.resolve_data_root_for_tree = boom
        root, kg_path, how = chp.resolve_roots(None)
        assert root == production_data_root(), how
        assert "production_data_root()" in how and "Boom" in how, how
        assert kg_path == kg_store_for_root(production_data_root())

        # Shape 2 — the worktree case, and the one a raise-only fallback misses: a
        # root that resolves cleanly and holds nothing.
        bare = tmp_path / "worktree-root"
        (bare / "_pipeline" / "vault-derived").mkdir(parents=True)
        chp.resolve_data_root_for_tree = lambda: bare
        root, kg_path, how = chp.resolve_roots(None)
        assert root == production_data_root(), \
            f"a resolved-but-storeless root reported {root}; the fallback did not fire"
        assert "holds no KG store" in how and str(bare) in how, how
        assert kg_path == kg_store_for_root(production_data_root())

        # The half that must not be rescued: a named root stays named, storeless or
        # not, so a sandboxed run can never quietly read the live KG.
        root, kg_path, how = chp.resolve_roots(str(bare))
        assert root == bare and how.startswith("--data-root"), (root, how)
        assert kg_path == kg_store_for_root(bare) and not kg_path.is_file()

        # And the environment route is a NAMED root too: `$LLOYD_DATA` pointing at an
        # empty directory must report that directory, not the live store.
        chp.resolve_data_root_for_tree = original
        monkeypatch.setenv("LLOYD_DATA", str(bare))
        root, _kg, how = chp.resolve_roots(None)
        assert root == bare, f"$LLOYD_DATA was overridden by {root}: {how}"
    finally:
        chp.resolve_data_root_for_tree = original


def test_the_probe_never_writes_the_production_data_root_from_a_fixture_run(tmp_path):
    """Every fixture run leaves the production `eval/baselines` as it found it.

    Clause 1's second half, measured rather than asserted about the code: the last
    round's fixture nodes put five artifacts into the live data root, because a run
    that resolves to production writes to production. `--out` and `--dry-run` are the
    two ways out, and this node fails if a future edit drops either.
    """
    from app.data_root import production_data_root

    baselines = production_data_root() / "eval" / "baselines"
    before = sorted(p.name for p in baselines.glob("complement-harvest-*.json")) \
        if baselines.is_dir() else []
    index, kg = _stores(tmp_path)
    _install_kg(tmp_path, kg)
    proc = _run(["--index", str(index), "--data-root", str(tmp_path),
                 "--out", str(tmp_path / "art.json")])
    assert proc.returncode == 0, proc.stderr
    after = sorted(p.name for p in baselines.glob("complement-harvest-*.json")) \
        if baselines.is_dir() else []
    assert after == before, f"the fixture run wrote into the live data root: {set(after) - set(before)}"
