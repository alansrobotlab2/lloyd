"""Groundskeeper retention sweep (scripts/groundskeeper/retention-sweep.py).

Pins the retention contract: old task logs are deleted, stale sessions are
gzipped (round-trip-validated, original removed), and anything younger than
the thresholds is untouched in both dry-run and apply modes.
"""
import gzip
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "groundskeeper" / "retention-sweep.py"


@pytest.fixture
def rs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retention_sweep", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect EVERY path this module can delete from. Patching only some of
    # them means a test that touches an unpatched root silently operates on real
    # data — sweep_autonomy_runs(apply=True) did exactly that and removed 844
    # live run records during development.
    for attr, sub in (
        ("TASKS_DIR", "tasks"),
        ("SESSIONS_DIR", "sessions"),
        ("AUTONOMY_RUNS_DIR", "autonomy-runs"),
        ("AUTONOMY_TASKS_DIR", "autonomy"),
        ("CANDIDATES_DIR", "skill-candidates"),
        ("TRANSCRIPT_SCRATCH_DIR", "transcript-scratch"),
    ):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, tmp_path / sub)
            (tmp_path / sub).mkdir(exist_ok=True)
    # Fail loudly if the module grows a new destructive root that this fixture
    # does not cover, rather than letting it reach the real filesystem.
    for name, value in list(vars(mod).items()):
        if name.endswith("_DIR") and isinstance(value, Path):
            assert tmp_path in value.parents or value == tmp_path, (
                f"{name} is not redirected into tmp_path (points at {value}) — "
                f"add it to the fixture before writing tests that touch it")
    return mod


def _backdate(path: Path, days: float) -> None:
    ts = time.time() - days * 86400
    os.utime(path, (ts, ts))


def _write_session(dir: Path, name: str, last_active_days: float,
                   platform: str | None = None) -> Path:
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=last_active_days)).isoformat()
    doc = {
        "session_id": name,
        "last_active": ts,
        "messages": [{"role": "user", "content": "hi"}],
    }
    if platform:
        doc["platform"] = platform
    p = dir / f"{name}.json"
    p.write_text(json.dumps(doc, indent=2))
    return p


def _write_spill(dir: Path, session_id: str, *, newest_days: float,
                 oldest_extra_days: float = 0.0, size: int = 1000) -> Path:
    """Create `<session_id>.tool-results/` holding spilled results whose NEWEST file is
    `newest_days` old, plus (optionally) an older one `oldest_extra_days` old.

    Returns the directory. The caller backdates nothing afterwards: mtime is set on the
    files only, which is the state the rung ages on — and the reason an `oldest_extra_days`
    argument exists at all. Backdating the directory too would leave the rule under
    `dir.stat().st_mtime` un-falsified.
    """
    d = dir / f"{session_id}.tool-results"
    d.mkdir(parents=True, exist_ok=True)
    (d / "toolu_newest.txt").write_text("n" * size)
    _backdate(d / "toolu_newest.txt", newest_days)
    if oldest_extra_days:
        (d / "toolu_oldest.txt").write_text("o" * size)
        _backdate(d / "toolu_oldest.txt", oldest_extra_days)
    return d


def test_old_task_logs_deleted_recent_kept(rs):
    now = time.time()
    old = rs.TASKS_DIR / "bg-old.log"
    old.write_text("x" * 100)
    _backdate(old, 45)
    recent = rs.TASKS_DIR / "bg-recent.log"
    recent.write_text("y" * 100)

    n, freed = rs.sweep_task_logs(apply=False, now=now)
    assert (n, freed) == (1, 100)
    assert old.exists()  # dry run touches nothing

    n, _ = rs.sweep_task_logs(apply=True, now=now)
    assert n == 1
    assert not old.exists()
    assert recent.exists()


def test_stale_session_gzipped_and_roundtrips(rs):
    now = time.time()
    stale = _write_session(rs.SESSIONS_DIR, "stale", last_active_days=120)
    fresh = _write_session(rs.SESSIONS_DIR, "fresh", last_active_days=5)

    n, _ = rs.sweep_sessions(apply=False, now=now)
    assert n == 1
    assert stale.exists()  # dry run touches nothing

    n, _ = rs.sweep_sessions(apply=True, now=now)
    assert n == 1
    assert not stale.exists()
    assert fresh.exists() and not fresh.with_suffix(".json.gz").exists()
    with gzip.open(rs.SESSIONS_DIR / "stale.json.gz", "rt") as fh:
        assert json.load(fh)["session_id"] == "stale"


def test_session_age_prefers_last_active_over_mtime(rs):
    # Recently-rewritten file (fresh mtime) with an old last_active must
    # still count as stale — mtime lies after reprocessing.
    p = _write_session(rs.SESSIONS_DIR, "rewritten", last_active_days=200)
    assert rs._session_age_days(p, time.time()) > 190


def test_run_records_age_by_frontmatter_not_mtime(rs, tmp_path):
    """A bulk operation on 2026-08-22 reset every run record's mtime, which made
    this sweep silently inert — 0 of 3,350 files matched. Age must come from the
    record's own frontmatter."""
    runs = tmp_path / "autonomy-runs" / "24"
    runs.mkdir(parents=True)
    monkey = rs.AUTONOMY_RUNS_DIR
    assert monkey  # fixture wired it

    old = runs / "run_24_20260329_120000.md"
    old.write_text("---\nrun_id: x\ncompleted_at: '2026-03-29T12:00:00+00:00'\n"
                   "status: success\n---\n\nbody\n")
    fresh = runs / "run_24_20260903_120000.md"
    fresh.write_text("---\nrun_id: y\ncompleted_at: '2026-09-03T12:00:00+00:00'\n"
                     "status: success\n---\n\nbody\n")
    # Both look brand-new on disk, exactly like the post-bulk-operation state.
    now = time.time()
    for p in (old, fresh):
        os.utime(p, (now, now))

    count, _ = rs.sweep_autonomy_runs(apply=False, now=now)
    assert count == 1, "the March record should be selected despite a fresh mtime"

    rs.sweep_autonomy_runs(apply=True, now=now)
    assert not old.exists()
    assert fresh.exists()


def test_legacy_epoch_named_records_are_swept(rs, tmp_path):
    """844 records from the pre-2026-03 scheduler were named <epoch_ms>.md and
    were permanently exempt: the glob only matched run_*.md."""
    runs = tmp_path / "autonomy-runs" / "37"
    runs.mkdir(parents=True)
    legacy = runs / "1774837994780.md"
    legacy.write_text("---\nrun_id: 1774837994780\n"
                      "started_at: '2026-03-30T01:40:05'\nstatus: success\n---\n\nx\n")
    keep = runs / "wiki-sweep-latest.json"
    keep.write_text("{}")
    now = time.time()
    os.utime(legacy, (now, now))

    count, _ = rs.sweep_autonomy_runs(apply=True, now=now)
    assert count == 1
    assert not legacy.exists()
    assert keep.exists(), "non-record files must be left alone"


def test_transcript_scratch_old_deleted_recent_kept(rs, capsys):
    """Backlog #566: the raw transcript scratch home is a bounded store. Measured on the
    live directory 2026-09-14: 5 files, 208,885 B, oldest mtime 2026-09-08 — nothing had
    ever been deleted, because nothing owned the directory. Deleting is licence the video
    note grants it (transcript_path + transcript_md5 ride along), not a guess.

    Every non-target here is backdated past the window, so each guard is falsifiable: an
    entry that is young can survive a sweep that has no guard at all, which is how a
    `recent.exists()` assertion starts checking nothing.

    Round SM_20260914_184202's review found the first draft still left one guard unfalsifiable,
    and it was right: `Path.unlink()` on a directory raises OSError, which the `except OSError`
    below catches and prints as a skip — so deleting `sweep_transcript_scratch`'s `is_file()`
    check changed neither the (1, 500) counts nor a `subdir.exists()`. Two things close that,
    both aimed at what a recursive or exception-fed implementation would actually do:
    a backdated file INSIDE the stray directory (a recursive sweep deletes it, and the byte
    count grows), and a check that the stray directory was never reported as a skip (a
    guard-less loop "survives" only by catching the EISDIR it caused, so it logs one)."""
    now = time.time()
    old = rs.TRANSCRIPT_SCRATCH_DIR / "T2v2pf_uypE.txt"
    old.write_text("t" * 500)
    _backdate(old, 45)
    recent = rs.TRANSCRIPT_SCRATCH_DIR / "1_8pzU44n-M.txt"
    recent.write_text("n" * 100)
    # A stray per-video subdir of the kind `/tmp/yt/0NvD6qNapiU/` was, aged past the window,
    # holding an aged file. Order matters: creating an entry inside a directory rewrites that
    # directory's mtime, so the inner file has to be written BEFORE the directory is backdated —
    # backdate first and the directory comes out young, which silently un-tests every guard
    # below (self-caught: the first draft did exactly that, and a mutation removing the sweep's
    # is_file() guard passed green).
    subdir = rs.TRANSCRIPT_SCRATCH_DIR / "0NvD6qNapiU"
    subdir.mkdir()
    # Out of reach of `iterdir()` — so it survives ONLY because the sweep does not walk. This is
    # the assertion that bites a recursive implementation; `subdir.exists()` alone could not,
    # because unlinking a directory raises and the sweep's OSError handler swallows it.
    stale_inside = subdir / "chunks.txt"
    stale_inside.write_text("c" * 700)
    _backdate(stale_inside, 45)
    _backdate(subdir, 45)
    outside = rs.TRANSCRIPT_SCRATCH_DIR.parent / "not-in-scratch.txt"
    outside.write_text("x")
    _backdate(outside, 45)
    link = rs.TRANSCRIPT_SCRATCH_DIR / "escape.txt"
    link.symlink_to(outside)

    n, freed = rs.sweep_transcript_scratch(apply=False, now=now)
    assert (n, freed) == (1, 500)
    assert old.exists(), "dry run must touch nothing"

    n, freed = rs.sweep_transcript_scratch(apply=True, now=now)
    # Exactly one file's worth of bytes: the dir, the symlink and the target outside it are
    # all past the window and none of them may count. This count is what pins the symlink
    # skip; the assertions below catch the variants that delete or follow anyway.
    assert (n, freed) == (1, 500)
    assert not old.exists()
    assert recent.exists(), "a transcript inside the window must survive"
    assert subdir.exists(), "a stray directory is not a transcript; leave it"
    assert stale_inside.exists(), "the sweep must not walk into a stray subdir and delete it"
    assert link.is_symlink(), "the sweep must skip a symlink, not unlink it"
    assert outside.exists(), "nothing outside the scratch dir may be unlinked through a link"

    # A second, independent net for the same defect, and the one that survives a change of
    # implementation: the only way the stray directory can reach the `except OSError` handler is
    # with the `is_file()` guard gone and `unlink()` raising EISDIR on it. An implementation that
    # counts only after a successful unlink keeps (1, 500) intact through that path — the counts
    # above then say nothing and this is what remains.
    # Verified to fire, 2026-09-14, because the first draft of this check passed green for the
    # wrong reason (see the backdating note above): with the guard removed the handler really does
    # print `! skip 0NvD6qNapiU: [Errno 21] Is a directory`, and the assertion below rejects it.
    # With the count assertions relaxed, that mutation fails on THIS assertion alone.
    logged = capsys.readouterr().err
    assert subdir.name not in logged, (
        f"sweep logged a skip for the stray directory — it is being rejected by an exception, "
        f"not by the is_file() guard:\n{logged}")


def test_transcript_scratch_missing_dir_is_zero(rs):
    """The scratch dir is created by an extraction session, not installed — first boot and
    every box that never extracted a video has no directory at all. That reports 0, it does
    not crash the sweep the rest of the stores depend on."""
    import shutil
    shutil.rmtree(rs.TRANSCRIPT_SCRATCH_DIR)
    assert rs.sweep_transcript_scratch(apply=True, now=time.time()) == (0, 0)


def test_dry_run_reports_the_transcript_store(rs, capsys, monkeypatch):
    """#566 clause 6's second half: the operator approves `--apply` from the dry run's printed
    line, so a store the sweep bounds but never prints is a store nobody can see being bounded.
    Run through `main()` with every deletable root redirected into tmp_path by the fixture, which
    is also what makes this safe to execute — the printed number has to come from the same call
    path an operator runs, not from a re-implementation of it."""
    stale = rs.TRANSCRIPT_SCRATCH_DIR / "cpqC9ib0-Kw.txt"
    stale.write_text("z" * 2048)
    _backdate(stale, 45)
    monkeypatch.setattr("sys.argv", ["retention-sweep.py"])

    assert rs.main() == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "transcript scratch" in ln]
    assert len(lines) == 1, f"expected exactly one transcript-scratch line, got {lines}"
    assert f">{rs.TRANSCRIPT_MAX_AGE_DAYS}d" in lines[0], lines[0]
    assert "1 deleted" in lines[0] and "2 KiB freed" in lines[0], \
        f"the dry run must report the store's count and bytes like the others: {lines[0]!r}"


# --------------------------------------------------------------------------------------
# The seventh rung: ~/lloyd/sessions/*.tool-results (backlog #1174).
#
# The store is the harness's spill sidecar: `app/harness/tool_result_spill.py` writes
# `<session_id>.tool-results/<tool_use_id>.{txt,json}` into the SAME directory rung 2
# bounds with `glob("*.json")`, so 130.1 MiB of the 544.3 MiB store (measured 2026-09-16)
# has been invisible to every age rule since the spill module landed. These cases run on
# tmp_path via the `rs` fixture, which already redirects SESSIONS_DIR — the rung deletes
# from that same root, so no new `*_DIR` had to be added to the fixture, and the fixture's
# loop at the top of this file would have failed for one that was not.
#
# Every non-target below is backdated PAST the window. That is deliberate and is the same
# discipline the transcript-scratch cases above follow: a younger control survives a rung
# that has no guard at all, so `survives.exists()` alone checks nothing.
# --------------------------------------------------------------------------------------


def test_spill_dir_older_than_the_window_is_deleted_and_younger_is_not(rs):
    """Clause 1: age is the NEWEST file inside the directory, against
    SPILL_MAX_AGE_DAYS = 30, and only --apply removes anything."""
    now = time.time()
    assert rs.SPILL_MAX_AGE_DAYS == 30

    # Over the window, every file in it: the delete case. 1000 + 512 B of it, so the
    # byte total is checkable rather than merely non-zero.
    gone = rs.SESSIONS_DIR / "over.tool-results"
    gone.mkdir()
    (gone / "toolu_a.txt").write_text("a" * 1000)
    (gone / "toolu_b.json").write_text("b" * 512)
    _backdate(gone / "toolu_a.txt", 45)
    _backdate(gone / "toolu_b.json", 40)

    # Under the window: must survive on its own merits, not on the guard's.
    keep_fresh = _write_spill(rs.SESSIONS_DIR, "under", newest_days=10, size=700)

    # The falsifier for "newest": a directory whose FIRST spill is over the window but
    # which a session has written into since. Aged on the oldest file, or on the
    # directory's own mtime (which is the age of the first file added to it), this one is
    # deleted — and it is the read-back target of a session that is still running.
    keep_mixed = _write_spill(rs.SESSIONS_DIR, "mixed", newest_days=5,
                              oldest_extra_days=90, size=800)

    n, freed = rs.sweep_session_spills(apply=False, now=now)
    assert (n, freed) == (1, 1512), \
        f"expected exactly the over-window dir and its exact bytes, got {(n, freed)}"
    assert gone.exists(), "dry run must touch nothing"

    n, freed = rs.sweep_session_spills(apply=True, now=now)
    assert (n, freed) == (1, 1512)
    assert not gone.exists(), "a dir whose newest spill is 45 d old must be reclaimed"
    assert keep_fresh.exists(), "a spill dir inside the window must survive"
    assert keep_mixed.exists(), (
        "a spill dir with a fresh file in it must survive even though it also holds a "
        "90-day-old spill — the rule is the NEWEST inner mtime")


def test_spill_dir_with_no_transcript_is_still_swept(rs):
    """Clause 2: the unjoinable id class. 372 of the 804 live spill dirs (61.4 MiB,
    measured 2026-09-16) name an id that has no `sessions/<id>.json` at all — task
    subagents, bench trials and the e2e harness spill under ids that never get a Lloyd
    transcript. Any rule of the form "the json is gone, so delete the siblings" cannot
    see them either, so the age rule must apply to them with nothing to join to.

    The three ids below are the live shapes, taken verbatim from the directory listing in
    the item: a task-subagent id, a bench trial id, an e2e id."""
    now = time.time()
    orphans = [
        _write_spill(rs.SESSIONS_DIR, "4743eb87c1f04de5a3f6be1e12c3f2a1", newest_days=31),
        _write_spill(rs.SESSIONS_DIR, "task:general-purpose:6ff2b13d", newest_days=40),
        _write_spill(rs.SESSIONS_DIR, "bench_baseline_1789237343_bench_006_06f26c32",
                     newest_days=60),
    ]
    for o in orphans:
        assert not (rs.SESSIONS_DIR / f"{o.name[:-len(rs.SPILL_DIR_SUFFIX)]}.json").exists(), \
            "the fixture must really be the unjoinable class, not a joinable one"
    # A sibling json belonging to some OTHER session proves the rung is not sweeping the
    # whole directory because it gave up on joining.
    other = _write_session(rs.SESSIONS_DIR, "some-session", last_active_days=1)

    n, freed = rs.sweep_session_spills(apply=False, now=now)
    assert n == 3, f"all three orphan classes must be candidates, got {n}"
    assert freed == 3 * 1000, f"one spill file's bytes per dir: {freed}"
    assert all(o.exists() for o in orphans), "dry run must touch nothing"

    n, freed = rs.sweep_session_spills(apply=True, now=now)
    assert (n, freed) == (3, 3000)
    for o in orphans:
        assert not o.exists(), f"{o.name} is over the window and joinable to nothing"
    assert other.exists(), "a transcript is not a spill dir; leave it alone"


def test_spill_dir_of_a_session_still_in_window_survives(rs):
    """Clause 3: the read-back guarantee. Spill exists precisely so the model can re-read
    a large tool result from disk with `Read` (`tool_result_spill.py:10-14`), so a
    directory that is over the sidecar window must NOT be taken while its own transcript
    is still inside that transcript's archive window. 30 < 90, so a long-running
    conversation reaches the spill window first and would otherwise lose the files its
    own prompt is pointing at."""
    now = time.time()
    # A conversation 60 d into its 90 d archive window, spill dir older than the 30 d sidecar
    # window: exactly the state a long-running chat reaches. This is the clause.
    spare = _write_spill(rs.SESSIONS_DIR, "conv", newest_days=45)
    _write_session(rs.SESSIONS_DIR, "conv", last_active_days=60)

    # Past its own archive line the same session gets no protection: the transcript is
    # about to leave the listings, so there is nothing left that can re-read the spill.
    archived = _write_spill(rs.SESSIONS_DIR, "oldconv", newest_days=45)
    _write_session(rs.SESSIONS_DIR, "oldconv", last_active_days=120)

    # And the window followed is the SESSION's own: a background run is archived at 30 d, so
    # a 40-day-old worker session is already past its line and its spill goes. Proves the
    # guard consults `_archive_age_for`, not a hard-coded 90.
    bg = _write_spill(rs.SESSIONS_DIR, "bgrun", newest_days=45)
    _write_session(rs.SESSIONS_DIR, "bgrun", last_active_days=40, platform="worker")

    n, freed = rs.sweep_session_spills(apply=False, now=now)
    assert n == 2, f"expected the archived conversation and the archived background run, got {n}"
    assert spare.exists() and archived.exists() and bg.exists()

    rs.sweep_session_spills(apply=True, now=now)
    assert spare.exists(), (
        "the read-back target of a session still inside its archive window must survive: "
        "the model's prompt points at this file by path")
    assert not archived.exists(), "a session past its archive line keeps no read-back right"
    assert not bg.exists(), (
        "a background run archived at 40 d follows its own 30 d window, not the 90 d "
        "conversation window")


def test_spill_dir_pattern_is_pinned_to_the_real_writer(tmp_path, monkeypatch):
    """Clause 4: the directory pattern must be pinned to `app.harness.tool_result_spill`,
    not to a string the sweep invented.

    This is the exact failure mode this script's own docstring records for run records —
    "a bulk operation on 2026-08-22 reset every run record's mtime, which made this sweep
    silently inert". A rung whose target drifts from the writer's reports `0 deleted` and
    reads as bounded. So the assertion is against the writer's own function, and on a
    module loaded with UNMODIFIED constants: patching SESSIONS_DIR into tmp_path and then
    reading the pattern off the patched module would compare the fixture against itself.

    No skip marker: a checkout that cannot import the writer has to fail here, because an
    unpinnable pattern is the regression under test, not a reason to pass.
    """
    import fnmatch
    import importlib.util

    unpatched = importlib.util.spec_from_file_location(
        "retention_sweep_unpatched", _SCRIPT)
    mod = importlib.util.module_from_spec(unpatched)
    unpatched.loader.exec_module(mod)  # module level only; nothing walks the filesystem

    from app.harness.tool_result_spill import _spill_dir

    sid = "20260917_040301_4743eb87"
    written = _spill_dir(sid)

    assert fnmatch.fnmatch(written.name, mod.SPILL_DIR_GLOB), (
        f"the writer puts spills at {written.name!r} but the sweep sweeps "
        f"{mod.SPILL_DIR_GLOB!r} — the rung is now silently inert")
    assert written.parent.name == mod.SESSIONS_DIR.name, (
        f"the writer spills into {written.parent} but the sweep walks "
        f"{mod.SESSIONS_DIR} — the rung is now silently inert")
    # The guard that spares an in-window session finds the transcript by stripping the
    # suffix off the directory name; if that no longer returns the session id, the guard
    # silently never matches and every live session's spill becomes a candidate.
    assert written.name[: -len(mod.SPILL_DIR_SUFFIX)] == sid, (
        "stripping SPILL_DIR_SUFFIX no longer recovers the session id, so the in-window "
        "guard cannot match anything")
    # And the boundary in the other direction: the change ledger's sidecar shares this
    # directory and shares the `<id>.<suffix>` shape, and must never match.
    assert not fnmatch.fnmatch(f"{sid}.changes", mod.SPILL_DIR_GLOB), \
        "the ledger's *.changes pre-images must not fall inside the spill glob"
    assert mod.SPILL_MAX_AGE_DAYS == 30


def test_empty_spill_dir_is_reclaimed_by_the_directorys_own_age(rs):
    """A spill dir with nothing in it has no inner mtime to age by, and holds nothing worth
    keeping; the directory is the only signal. Untested, this branch is where an
    '0 candidates' reading of a partly-drained store would hide."""
    now = time.time()
    empty = rs.SESSIONS_DIR / "emptied.tool-results"
    empty.mkdir()
    _backdate(empty, 45)
    young_empty = rs.SESSIONS_DIR / "just-spilled.tool-results"
    young_empty.mkdir()

    assert rs.sweep_session_spills(apply=True, now=now) == (1, 0)
    assert not empty.exists(), "an empty spill dir older than the window is dead weight"
    assert young_empty.exists()


def test_spill_store_is_reported_and_apply_removes_what_dry_run_counted(
        rs, capsys, monkeypatch):
    """Clause 5: `--apply` and dry-run print the same line with the same numbers, and the
    dry run removes nothing — so a reported `0 deleted` can only mean "nothing over
    window", never "no rule", which is how the old `run_*.md` glob sat inert over 3,350
    records. Run through `main()`, because the operator approves `--apply` from the line
    this call path prints, not from a re-implementation of it.

    The `.changes` sibling is in here too: same parent, same `<id>.<suffix>` shape, and
    owned by the change ledger's own `prune()` — a sweep that reached it would delete the
    pre-images the ledger exists to protect."""
    stale = _write_spill(rs.SESSIONS_DIR, "task:general-purpose:6ff2b13d", newest_days=45,
                         size=2048)
    fresh = _write_spill(rs.SESSIONS_DIR, "still-running", newest_days=2, size=4096)
    changes = rs.SESSIONS_DIR / "some-session.changes"
    changes.mkdir()
    (changes / "turn-1").write_text("pre-image" * 200)
    _backdate(changes / "turn-1", 45)
    _backdate(changes, 45)

    def spill_line() -> str:
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "spill" in ln]
        assert len(lines) == 1, f"expected exactly one spill line, got {lines}"
        return lines[0]

    monkeypatch.setattr("sys.argv", ["retention-sweep.py"])
    assert rs.main() == 0
    dry = spill_line()
    assert f">{rs.SPILL_MAX_AGE_DAYS}d" in dry, dry
    assert "1 deleted" in dry and "2 KiB freed" in dry, \
        f"the dry run must report the store's count and bytes like the other six: {dry!r}"
    assert stale.exists(), "dry run must remove nothing"

    monkeypatch.setattr("sys.argv", ["retention-sweep.py", "--apply"])
    assert rs.main() == 0
    applied = spill_line()
    assert "1 deleted" in applied and "2 KiB freed" in applied, \
        f"--apply must report the same counts the dry run promised: {applied!r}"
    assert not stale.exists()
    assert fresh.exists()
    assert changes.exists() and (changes / "turn-1").exists(), \
        "the change ledger's pre-images are its own to prune; the sweep must not reach them"

    # And the line goes to 0 on a drained store — the number an operator sees on the run
    # after this one, which is the difference between an inert rule and a finished one.
    monkeypatch.setattr("sys.argv", ["retention-sweep.py"])
    assert rs.main() == 0
    assert "0 deleted" in spill_line(), "a drained store must report 0, and mean it"
