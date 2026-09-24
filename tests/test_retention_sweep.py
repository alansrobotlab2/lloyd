"""Groundskeeper retention sweep (scripts/groundskeeper/retention-sweep.py).

Pins the retention contract: old task logs are deleted, stale sessions are
gzipped (round-trip-validated, original removed), and anything younger than
the thresholds is untouched in both dry-run and apply modes.

Pins the root those stores live under, too — which root a `--apply` reaches is a
retention contract like any other, because the wrong answer here is a real sweep
of the wrong tree (`#1415`); see the section at the bottom of this file.
"""
import gzip
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

import app.data_root as dr
import app.paths as paths

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "groundskeeper" / "retention-sweep.py"


#: Module-level `Path` constants that are not delete targets, so they may stay where
#: the module resolved them, each with the reason:
#:
#:   DATA_ROOT  the root every other store is derived from, pinned against `app.paths`
#:              by name below; the sweep removes things inside it, never the root
#:   _TREE      the checkout the script itself lives in, used only to find `app/` on
#:              sys.path — it is never passed to a delete

#: Anything else UPPER_CASE-ish and a `Path` is assumed to be something a test could
#: delete from until it says otherwise here. (The name test is deliberately case-blind:
#: a `_FOO` private constant holding a path is just as deletable as a public one.)
NON_TARGET_PATHS = frozenset({"DATA_ROOT", "_TREE"})


def _unredirected_destructive_paths(mod, keep_under: Path,
                                    exempt=NON_TARGET_PATHS) -> list[str]:
    """Names of `mod`'s module-level path constants that do not live under `keep_under`.

    The rule is "every UPPER_CASE module constant that is a `Path`", not a list of name
    suffixes. A suffix rule (`_DIR`, then `_DIR` + `_DB`) is only as wide as the last
    name somebody thought of: `WORKERS_DB_FILE`, `QUEUE_DB` or `SPILL_ROOT` would each
    have reached a live store with this guard nodding, because a suffix records how a
    constant is *named*, not what it *points at*. A constant that genuinely is not a
    delete target has to be exempted by name, which turns each such gap into a decision
    somebody wrote down.

    One predicate, used by the fixture below to fail loudly and by
    `test_the_guard_names_any_unredirected_path_constant` to prove it names a file
    constant and a name that does not exist in the script yet. A second copy written
    inside the test would pass whatever the fixture did, which is the thing asserted.
    """
    return sorted(
        name for name, value in vars(mod).items()
        if isinstance(value, Path) and not name.startswith("__") and name not in exempt
        and not (value == keep_under or keep_under in value.parents)
    )


@pytest.fixture
def rs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("retention_sweep", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Redirect EVERY path this module can delete from. Patching only some of
    # them means a test that touches an unpatched root silently operates on real
    # data — sweep_autonomy_runs(apply=True) did exactly that and removed 844
    # live run records during development.
    for attr, sub, make_dir in (
        ("TASKS_DIR", "tasks", True),
        ("SESSIONS_DIR", "sessions", True),
        ("AUTONOMY_RUNS_DIR", "autonomy-runs", True),
        ("AUTONOMY_TASKS_DIR", "autonomy", True),
        ("CANDIDATES_DIR", "skill-candidates", True),
        ("TRANSCRIPT_SCRATCH_DIR", "transcript-scratch", True),
        # A database FILE, not a directory — and deliberately not created. The
        # absent-file path is both the state a fresh install, a sandbox and a
        # round's tree are in and the case clause 1 must not turn into a schema
        # repair; a `mkdir` here would hand every test a directory where a
        # database should be, which sqlite reports as "unable to open".
        ("WORKERS_DB", "workers.db", False),
    ):
        if hasattr(mod, attr):
            monkeypatch.setattr(mod, attr, tmp_path / sub)
            if make_dir:
                (tmp_path / sub).mkdir(exist_ok=True)
    # Fail loudly if the module grows a new destructive root that this fixture
    # does not cover, rather than letting it reach the real filesystem.
    unredirected = _unredirected_destructive_paths(mod, tmp_path)
    assert not unredirected, (
        f"{unredirected} not redirected into tmp_path — add them to the fixture "
        f"before writing tests that touch them")
    # The patching above says only that each dir is patchable. Before the patch,
    # the module resolved a root for itself; every destructive test in this file
    # runs against the module that resolved it, so pin what it resolved: the
    # scratch root this process exported, identical to `app.paths`' answer for the
    # same tree, and not the machine's live root (#1415).
    assert mod.DATA_ROOT == paths.DATA_ROOT, (
        f"the sweep resolved {mod.DATA_ROOT}, app.paths resolved {paths.DATA_ROOT}")
    assert mod.DATA_ROOT != paths.PRODUCTION_DATA_ROOT
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


# --------------------------------------------------------------------------------------
# Which root the sweep resolves (backlog #1415)
#
# Every test above reaches the stores through the `rs` fixture, which patches the
# directories into `tmp_path`. That is the right shape for testing what gets
# deleted and the wrong shape for testing WHERE: the fixture supplies the paths, so
# it cannot observe the resolution that produced them — and the resolution is the
# thing that was wrong.
#
# The script used to resolve the root itself as `${LLOYD_DATA:-~/lloyd-data}`: rule
# 2 of `app.paths`' three rules with neither the marker check nor rule 3.
# `LLOYD_DATA` is exported by the gate, the canary and the test suite, and by
# nothing in production (nothing should export it, and nothing does), so an unset
# variable is the normal condition of every shell — including a sandbox's and a
# round worktree's — and the answer from ANY tree was `<account home>/lloyd-data`:
# the marked live root, 56,272 files on 2026-09-23, which is precisely what
# `sweep_sessions` and `sweep_session_spills` unlink. Nothing downstream would have
# caught it either: the harness's delete guard parses Bash command strings and
# never sees a Python `unlink`, and `vaultwatch`'s tripwire needs 10 % AND 200 files
# gone inside 900 s — roughly 5,600 removals from a root that size.
#
# The fix is that the script imports the rules instead of restating them, so these
# cases load the module fresh with the resolution's inputs under control: the
# environment variable, the tree the script lives in, and whether the account's
# production root carries its marker.
# --------------------------------------------------------------------------------------

#: The checkout the script resolves itself in — `parents[2]` of
#: `scripts/groundskeeper/retention-sweep.py`, the same arithmetic the script uses.
_SWEEP_TREE = _SCRIPT.resolve().parents[2]


def _load(monkeypatch, *, lloyd_data=None, production_root=None, marker=False):
    """Import the script fresh, with the inputs the resolution reads controlled.

    `lloyd_data` is the value for `LLOYD_DATA`; leaving it None takes the variable
    away, which is the state of every production shell and the state the three
    rules turn on. `production_root` makes the tree the script lives in be the
    production checkout for the duration of the load, with its
    `<account home>/lloyd-data` standing at that path — nothing else on this machine
    can be the production checkout, and what is on trial is the rule, not this box.
    Two inputs say "production" and both are pinned, because both are inputs on
    purpose: the tree matching the live checkout, AND that tree not being a linked
    worktree — which is what keeps a worktree cut off the live path on its own data.
    This suite runs inside exactly such a worktree, so leaving the second one real
    would answer rule 3 no matter what the first said. `marker` says whether that
    root carries `.lloyd-data-root`.
    """
    if lloyd_data is None:
        monkeypatch.delenv("LLOYD_DATA", raising=False)
    else:
        monkeypatch.setenv("LLOYD_DATA", str(lloyd_data))
    if production_root is not None:
        prod = Path(production_root)
        if marker:
            (prod / dr.DATA_ROOT_MARKER).write_text("{}")
        monkeypatch.setattr(dr, "live_checkout", lambda: _SWEEP_TREE)
        monkeypatch.setattr(dr, "tree_is_worktree", lambda tree: False)
        monkeypatch.setattr(dr, "production_data_root", lambda: prod)
    spec = importlib.util.spec_from_file_location("retention_sweep_fresh", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_unset_in_a_non_production_tree_resolves_inside_that_tree(monkeypatch):
    """Clause 1: with the variable absent and a tree that is not the production
    checkout, the root is `<tree>/.lloyd-data` — the sandbox or the round sweeps its
    own data — and never `<account home>/lloyd-data`.

    The tree here is a real checkout on disk, because the script is loaded from its
    own path rather than from a fixture: which tree a script lives in is a fact about
    it, and a test that passes a fake tree would pin the resolver, not the script.
    """
    assert (_SWEEP_TREE != dr.live_checkout().resolve()
            or dr.tree_is_worktree(_SWEEP_TREE)), (
        "this suite is running from the production checkout, where rule 2 is the"
        " right answer and clause 1 has nothing to falsify here")
    root_in_tree = _SWEEP_TREE / ".lloyd-data"
    existed_before = root_in_tree.exists()

    mod = _load(monkeypatch)

    assert mod.DATA_ROOT == root_in_tree
    assert mod.DATA_ROOT.is_relative_to(_SWEEP_TREE)
    assert not mod.DATA_ROOT.is_relative_to(dr.PRODUCTION_DATA_ROOT)
    # The store this script unlinks, checked on its own: a root that is right while
    # the dirs are derived from somewhere else is the bug all over again.
    assert mod.SESSIONS_DIR == root_in_tree / "sessions"
    # Importing must leave the filesystem as it found it. `app.paths` does create
    # `SESSIONS_DIR` at import; this module must not, or a mere import outside the
    # venv starts a second data tree.
    assert root_in_tree.exists() is existed_before, (
        "importing the sweep created the root it resolved")


def test_the_production_checkout_still_resolves_to_the_account_root(tmp_path, monkeypatch):
    """Clause 2: with the variable absent and the production tree whose root carries
    its marker, the same resolution returns `<account home>/lloyd-data`.

    This is the weekly run's exact state: autonomy task #79's skill is
    `~/lloyd/.venvs/lloyd/bin/python ~/lloyd/scripts/groundskeeper/retention-sweep.py
    --apply` — the production checkout, nothing exported, so rule 2. A red here
    would not mean the sweep had been made safe; it would mean the sweep had become
    a no-op reporting `0 deleted` against an empty `.lloyd-data/` while the live
    stores keep growing.
    """
    prod = tmp_path / "home" / "lloyd-data"
    prod.mkdir(parents=True)
    # What the stand-in stands in for: the second rule's answer is the passwd
    # home's `lloyd-data`, which is what `production_data_root()` is.
    assert dr.PRODUCTION_DATA_ROOT == dr.ACCOUNT_HOME / "lloyd-data"

    mod = _load(monkeypatch, production_root=prod, marker=True)

    assert mod.DATA_ROOT == prod
    assert mod.SESSIONS_DIR == prod / "sessions"
    assert mod.TASKS_DIR == prod / "_pipeline" / "tasks"
    assert mod.AUTONOMY_RUNS_DIR == prod / "autonomy-runs"
    assert mod.TRANSCRIPT_SCRATCH_DIR == prod / "_pipeline" / "tmp"


def test_production_root_without_the_marker_refuses_before_any_delete(tmp_path, monkeypatch,
                                                                     capsys):
    """Clause 3: a production checkout whose root has lost `.lloyd-data-root` is
    refused, with a non-zero exit, before a single store is opened.

    Two wrong answers are on either side of this one. Falling back to the tree
    starts the second copy of everything inside the code checkout that the whole
    data move exists to prevent; falling back to an empty `.lloyd-data` and reporting
    `0 deleted` is worse than refusing, because it looks like a clean sweep. The
    refusal has to be the resolution's own — if a later change caught
    `DataRootMissing` and carried on, `pytest.raises` here is what fails.
    """
    prod = tmp_path / "home" / "lloyd-data"
    (prod / "sessions").mkdir(parents=True)
    stale = prod / "sessions" / "20260101_000000_oldsession.json"
    stale.write_text(json.dumps({"session_id": "oldsession",
                                 "last_active": "2026-01-01T00:00:00"}))
    _backdate(stale, 400)          # far past the 90 d archive window

    with pytest.raises(SystemExit) as excinfo:
        _load(monkeypatch, production_root=prod)      # marker absent
    assert excinfo.value.code == 2, "refusal must be a non-zero exit, not a traceback"

    err = capsys.readouterr().err
    assert "REFUSING" in err, f"the sweep must say it refused: {err!r}"
    assert dr.DATA_ROOT_MARKER in err, f"and name what is missing: {err!r}"
    assert stale.exists(), "the refusal precedes any store being opened"


def test_the_resolved_root_is_printed_in_both_modes(rs, capsys, monkeypatch):
    """Clause 4: the dry run and `--apply` both print the root their numbers
    describe.

    The operator approves `--apply` from the dry run's eight counts, and every one
    of them is a count of whatever root this run resolved — a value that depends on
    the caller's tree, its environment and its `$HOME`, and appeared nowhere in the
    report until #1415. Through `main()`, like the two reporting tests above: the
    line has to come out of the path an operator runs, not a re-implementation of it.
    """
    def root_line() -> str:
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "data root" in ln]
        assert len(lines) == 1, f"expected exactly one data-root line, got {lines}"
        return lines[0]

    monkeypatch.setattr("sys.argv", ["retention-sweep.py"])
    assert rs.main() == 0
    dry = root_line()
    assert str(rs.DATA_ROOT) in dry, f"a dry run must name the root it counted: {dry!r}"

    monkeypatch.setattr("sys.argv", ["retention-sweep.py", "--apply"])
    assert rs.main() == 0
    applied = root_line()
    assert str(rs.DATA_ROOT) in applied, f"--apply must name the root it deleted from: {applied!r}"
    assert applied == dry, "both modes must label their numbers with the same root"


def test_a_bare_interpreter_reports_the_root_it_resolved(tmp_path):
    """The same claim across the boundary that matters: cron and autonomy task #79
    run this file as a program, so the printed root and the exit status are the ones
    a shell sees — and with `python3`, not the project venv, which is the whole
    reason `app/data_root.py` has to be stdlib-only.

    Run from this checkout, which is not the production one, with `LLOYD_DATA`
    removed from the child's environment: the answer must be `<tree>/.lloyd-data`.
    A dry run writes nothing, and `LLOYD_VAULT_ROOT` keeps the one vault-touching
    rung pointed at a directory that does not exist.
    """
    python = shutil.which("python3")
    assert python, "the stdlib-only claim is a claim about a bare interpreter"
    env = {k: v for k, v in os.environ.items() if k != "LLOYD_DATA"}
    env["LLOYD_VAULT_ROOT"] = str(tmp_path / "no-vault-here")

    proc = subprocess.run([python, str(_SCRIPT)], env=env, cwd=str(_SWEEP_TREE),
                          capture_output=True, text=True, timeout=180)

    assert proc.returncode == 0, f"bare python3 could not run the sweep: {proc.stderr}"
    assert "DRY RUN" in proc.stdout, proc.stdout
    assert f"data root: {_SWEEP_TREE / '.lloyd-data'}" in proc.stdout, \
        f"the invocation resolved a different root than rule 3 gives it:\n{proc.stdout}"
    assert str(dr.PRODUCTION_DATA_ROOT) not in proc.stdout, \
        "a non-production invocation named the live root as its target"


def test_the_one_resolution_owns_every_directory_the_sweep_touches(tmp_path, monkeypatch):
    """Clause 5: `LLOYD_DATA` at a scratch root puts `DATA_ROOT` and every
    module-level `*_DIR` beneath it.

    The `rs` fixture patches each dir and then fails on an unpatched one, which
    catches a new root escaping the fixture but cannot catch the six drifting apart,
    because the fixture is what places them. This loads the module with nothing
    patched and reads where its own arithmetic put them — the only way a
    `SOMETHING_DIR = Path.home() / ...` added beside the others becomes a test
    failure rather than a live-data incident. `AUTONOMY_TASKS_DIR` is in the set on
    purpose: it is the one store outside the data root (the vault's task files), and
    it lands under the scratch root here only because the test sends
    `LLOYD_VAULT_ROOT` there, which is the knob that keeps a round's `--apply` out of
    the live activity logs.
    """
    root = tmp_path / "scratch-root"
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(root / "vault"))

    mod = _load(monkeypatch, lloyd_data=root)

    assert mod.DATA_ROOT == root
    swept = {name: value for name, value in vars(mod).items()
             if name.endswith("_DIR") and isinstance(value, Path)}
    assert set(swept) == {"AUTONOMY_RUNS_DIR", "AUTONOMY_TASKS_DIR", "CANDIDATES_DIR",
                          "SESSIONS_DIR", "TASKS_DIR", "TRANSCRIPT_SCRATCH_DIR"}, \
        f"the sweep gained or lost a store dir; update this set deliberately: {sorted(swept)}"
    for name, value in sorted(swept.items()):
        assert value.is_relative_to(root), f"{name} = {value} is outside the root {root}"
    assert swept["AUTONOMY_TASKS_DIR"] == root / "vault" / "autonomy"


def test_the_vault_rung_reads_the_override_the_guardian_already_reads(tmp_path, monkeypatch):
    """The sweep's one write outside the data root used to be
    `Path.home() / "obsidian" / "autonomy"` with no knob at all. Under a gate's HOME
    that is the round's `~/obsidian`, which `scripts/automod/worktree.py`
    `ensure_round_home` links into the live vault — `HOME_LINK_SKIP` covers `lloyd`
    and `lloyd-data`, not `obsidian` — and the same file says the symlink exists to
    survive `rmtree`, which does nothing about `write_text`. So `--apply` in a round
    pruned the LIVE activity logs. The override is not a new invention: it is
    `LLOYD_VAULT_ROOT`, the name `vaultwatch.py:46` and `scripts/backup/
    backup-vault.sh:28` already read. With it unset the default must still equal
    `app.paths.VAULT_ROOT`, or the sweep would bound a different vault than the
    system writes to.
    """
    monkeypatch.delenv("LLOYD_VAULT_ROOT", raising=False)
    assert dr.vault_root() == paths.VAULT_ROOT
    mod = _load(monkeypatch, lloyd_data=tmp_path / "root")
    assert mod.AUTONOMY_TASKS_DIR == paths.VAULT_ROOT / "autonomy"

    copy = tmp_path / "vault-copy"
    monkeypatch.setenv("LLOYD_VAULT_ROOT", str(copy))
    assert dr.vault_root() == copy
    redirected = _load(monkeypatch, lloyd_data=tmp_path / "root")
    assert redirected.AUTONOMY_TASKS_DIR == copy / "autonomy"


def test_the_sweep_imports_the_resolver_instead_of_restating_it(monkeypatch):
    """The shape of #1415 was never a wrong constant; it was a second
    implementation of the rules, one copy per stdlib job. Pinning the outcome
    without pinning the sharing would let the next change fix this script and leave
    `agent-services/guardian/{policy,datawatch}.py`, `idle-worker.py`,
    `livekit_worker.py` and the two backup scripts holding their own copies — the
    list `architecture/data-home.md` still names as unresolved.

    The three names the script imports must be the same objects `app.data_root`
    exports, `app.paths` must export them too, and the old formula must not come
    back as a string in the file.
    """
    mod = _load(monkeypatch, lloyd_data=Path("/tmp/whatever-root"))

    assert mod.resolve_data_root_for_tree is dr.resolve_data_root_for_tree
    assert mod.DataRootMissing is dr.DataRootMissing
    assert mod.vault_root is dr.vault_root
    # One implementation behind both spellings is the half that makes a fix here a
    # fix for the system, rather than for one script.
    assert paths.resolve_data_root is dr.resolve_data_root
    assert paths.data_root_for_tree is dr.data_root_for_tree
    assert paths.production_data_root is dr.production_data_root
    assert paths.DataRootMissing is dr.DataRootMissing
    assert paths.DATA_ROOT_MARKER == dr.DATA_ROOT_MARKER

    src = _SCRIPT.read_text(encoding="utf-8")
    assert 'Path.home() / "lloyd-data"' not in src, (
        "the second implementation of rule 2 is back in the script that deletes")


# ---------------------------------------------------------------------------
# The eighth rung: workers.db table `runs` (backlog #1018).
#
# The queue appended a row per run and nothing in this repository ever deleted
# one — `DELETE FROM runs` appears in no commit in its history, so the store the
# architecture doc called unbounded stayed unbounded. These tests seed a real
# sqlite database holding the queue's real schema and drive both the rung and the
# CLI that reports it, because every prior claim about this store was made from
# the other side: from the doc, or from a dry-run line that printed nothing.
# ---------------------------------------------------------------------------

#: (run_id, age in days). 31 is one day past the horizon, 29 is one day inside it;
#: the pair is the boundary, and a rule that got the comparison backwards would
#: keep the first and delete the second.
_RUNS_ROWS = (("run-past", 31.0), ("run-inside", 29.0), ("run-today", 0.0))


def _seed_runs_db(path: Path, rows=_RUNS_ROWS) -> Path:
    """Create a database carrying the queue's own schema, with `rows` in `runs`.

    The schema is `workers.queue._SCHEMA` — the same string `WorkQueue` runs
    through `executescript` — not a hand-written lookalike. A `runs` table
    invented here would let this whole section pass against a database whose
    `completed_at` had been renamed, dropped, or made nullable, and the column
    being `TEXT NOT NULL` is load-bearing: the horizon compares ISO strings, and
    a NULL would sort before all of them.
    """
    from datetime import datetime, timedelta, timezone

    from workers.queue import _SCHEMA

    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    now = datetime.now(timezone.utc)
    for run_id, age_days in rows:
        ts = (now - timedelta(days=age_days)).isoformat()
        conn.execute(
            "INSERT INTO runs (run_id, source, status, started_at, completed_at,"
            " duration_seconds, summary) VALUES (?,?,?,?,?,?,?)",
            (run_id, "scheduled-task", "success", ts, ts, 1.0, run_id),
        )
    conn.commit()
    conn.close()
    return path


def _store_line(stdout: str, store: str) -> str:
    """The one report line naming `store`, from a captured `main()` stdout.

    Asserting exactly one line rather than the first match: a store reported twice
    means two different numbers on screen for the same table, which is the failure
    mode an operator reading a dry run cannot see.
    """
    lines = [ln for ln in stdout.splitlines() if store in ln]
    assert len(lines) == 1, f"expected one {store} line, got {lines}"
    return lines[0]


def _run_ids(path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT run_id FROM runs")}
    finally:
        conn.close()


def test_runs_older_than_the_horizon_are_deleted_and_younger_kept(rs):
    """#1018 clause 1: one `--apply` deletes every row past `WORKER_RUN_MAX_AGE_DAYS` and
    leaves a row completed 29 days ago untouched.

    `WORKER_RUN_MAX_AGE_DAYS` is asserted to be 30 rather than assumed: the horizon
    is a policy the operator reads off the report line and the skill's table, so a
    constant that drifted would drift the report with it and stay self-consistent.
    """
    assert rs.WORKER_RUN_MAX_AGE_DAYS == 30
    path = _seed_runs_db(rs.WORKERS_DB)
    now = time.time()

    rows, freed, skip = rs.sweep_worker_runs(apply=False, now=now, db=path)
    assert (rows, freed, skip) == (1, 0, "")
    assert _run_ids(path) == {"run-past", "run-inside", "run-today"}, (
        "a dry run must not remove a row it counted")

    rows, freed, skip = rs.sweep_worker_runs(apply=True, now=now, db=path)
    assert (rows, skip) == (1, "")
    assert _run_ids(path) == {"run-inside", "run-today"}


def test_dry_run_and_apply_print_the_same_workers_store_line(rs, capsys, monkeypatch):
    """#1018 clause 2: dry-run deletes nothing and prints the store with its `>30d`
    horizon in the form the other store lines use, and `--apply` prints the same
    numbers it promised.

    Run through `main()`, which is what task #79 and cron invoke: the operator
    approves `--apply` from the dry run's eight lines, so a line that only exists
    in one mode, or counts differently in the two, is a number nobody can approve
    against.
    """
    path = _seed_runs_db(rs.WORKERS_DB)

    def store_line() -> str:
        lines = [ln for ln in capsys.readouterr().out.splitlines() if "workers.db" in ln]
        assert len(lines) == 1, f"expected exactly one workers.db line, got {lines}"
        return lines[0]

    monkeypatch.setattr("sys.argv", ["retention-sweep.py"])
    assert rs.main() == 0, "dry run must exit 0"
    dry = store_line()
    assert f">{rs.WORKER_RUN_MAX_AGE_DAYS}d" in dry, dry
    assert "1 deleted" in dry, f"dry run must count the row the apply will remove: {dry!r}"
    assert _run_ids(path) == {"run-past", "run-inside", "run-today"}

    monkeypatch.setattr("sys.argv", ["retention-sweep.py", "--apply"])
    assert rs.main() == 0, "--apply must exit 0"
    applied = store_line()
    assert "1 deleted" in applied, (
        f"--apply must report what it removed, not restate the dry run: {applied!r}")
    assert _run_ids(path) == {"run-inside", "run-today"}


def test_the_store_is_reached_through_the_queue_configured_path(monkeypatch):
    """Clause 3, first half: the module's one database constant equals
    `workers.queue.configured_db_path()`.

    Loaded unpatched, on purpose — the `rs` fixture redirects `WORKERS_DB`
    precisely so tests cannot reach the live queue, which also makes it unable to
    say what the constant would have been. This is the resolution the sweep runs
    with, and the reason it is an import rather than a literal: a second rule
    here would prune a file the queue never writes and report success about it.
    """
    from workers.queue import configured_db_path

    mod = _load(monkeypatch)
    assert mod.WORKERS_DB == Path(configured_db_path())


def test_the_path_fallback_when_the_queue_module_cannot_be_imported(rs, monkeypatch):
    """Clause 3, second half: with no venv the constant is `DATA_ROOT / "workers.db"`.

    Cron and autonomy task #79 run this script with a bare `python3`, and
    `workers.queue` pulls in `app.config`, which needs the project venv — which is
    why `app/data_root.py` is stdlib-only (#1415). So the import failing is the
    normal case for the one invocation that prunes real rows, and its answer has
    to be the same file, reached through the resolver this module already trusts.
    """
    monkeypatch.setitem(sys.modules, "workers.queue", None)
    assert rs._resolve_workers_db() == rs.DATA_ROOT / "workers.db"


#: Run by a child process: take the write lock and hold it open, uncommitted, until
#: stdin closes. `BEGIN IMMEDIATE` plus an uncommitted INSERT is what `WorkQueue` does
#: from the backend, so the contention is the real one — and it crosses a process
#: boundary, which two connections inside one interpreter cannot guarantee: they share
#: the same sqlite handle table and one process's locking quirks.
_LOCK_HOLDER = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], timeout=30)
conn.execute("BEGIN IMMEDIATE")
conn.execute("INSERT INTO runs (run_id, source, status, started_at, completed_at)"
             " VALUES ('holder', 'scheduled-task', 'success', 'x', 'x')")
print("HELD", flush=True)
sys.stdin.readline()
conn.rollback()
"""


@pytest.fixture
def db_write_lock_holder():
    """Hold a write transaction open on a database from a SECOND PROCESS."""
    procs = []

    def hold(db_path):
        proc = subprocess.Popen(
            [sys.executable, "-c", _LOCK_HOLDER, str(db_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        first = proc.stdout.readline().strip()
        assert first == "HELD", f"holder child never took the write lock: {first!r}"
        procs.append(proc)
        return proc

    yield hold
    for proc in procs:
        proc.stdin.close()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:      # never leave a lock holder running
            proc.kill()
            proc.wait(timeout=20)


def test_a_database_the_queue_holds_open_is_skipped_not_a_failure(rs, db_write_lock_holder):
    """#1018 clause 4: a db another PROCESS holds is a skip with a bounded wait.

    The contention that matters is the backend's queue pool writing the live database
    while the weekly sweep runs — two processes, no shared state — so the lock is held
    by a child process, not by a second connection in this interpreter. The wait is
    timed as well as checked: the clause's whole promise is that the sweep yields in
    about `DB_BUSY_TIMEOUT_MS` rather than hanging the job that is supposed to report
    the skip, and a `busy_timeout_ms` nobody asserts is a constant that can silently
    drift back to blocking.
    """
    assert rs.DB_BUSY_TIMEOUT_MS == 5000, (
        "the bound is a 5 s yield to the live queue; the elapsed-time assertion below "
        "is written against this constant, so it has to be pinned, not assumed")
    path = _seed_runs_db(rs.WORKERS_DB)
    db_write_lock_holder(path)

    started = time.monotonic()
    rows, freed, skip = rs.sweep_worker_runs(apply=True, now=time.time(), db=path)
    waited = time.monotonic() - started

    assert (rows, freed) == (0, 0)
    assert skip.startswith("SKIPPED (database locked"), (
        f"a locked store must report the reason the skill documents: {skip!r}")
    assert 4.5 <= waited < 60, (
        f"waited {waited:.2f}s — the default must be one bounded wait of about "
        f"{rs.DB_BUSY_TIMEOUT_MS} ms, not a no-try and not a hang")
    assert "run-past" in _run_ids(path), "a skip must not have deleted anything"


def test_main_reports_the_skip_and_still_exits_zero(rs, capsys, monkeypatch,
                                                   db_write_lock_holder):
    """Clause 4, across the process boundary the operator sees: `main()` reports
    `SKIPPED` on the store line and returns 0.

    Exit status is the contract task #79's run record is graded on — a non-zero
    exit makes the weekly job fail and hides which store could not be swept, while
    exit 0 with the word on the line says both that the job ran and that this
    particular store is still unbounded. `SKIPPED` replaces the counts rather than
    sitting beside a `0`, because the skill tells the operator that a `0` on this
    store means the window genuinely held nothing.
    """
    path = _seed_runs_db(rs.WORKERS_DB)
    db_write_lock_holder(path)
    monkeypatch.setattr("sys.argv", ["retention-sweep.py", "--apply"])
    assert rs.main() == 0, "a locked store must not fail the whole sweep"

    lines = [ln for ln in capsys.readouterr().out.splitlines() if "workers.db" in ln]
    assert len(lines) == 1, lines
    assert "SKIPPED" in lines[0], lines[0]
    assert "database locked" in lines[0], lines[0]
    assert "0 deleted" not in lines[0], (
        f"a skip must not read as an empty window: {lines[0]!r}")
    assert "run-past" in _run_ids(path)


def test_an_absent_database_or_table_is_zero_deleted_not_a_repair(rs):
    """An absent db, and a db with no `runs` table, are both `0 deleted` with no skip.

    The acceptance names this explicitly to keep the change small: the sweep is a
    cleanup tool, and schema creation or migration belongs to `WorkQueue` — a
    cleanup job that repairs schema is a second owner of the schema, which is how
    the store got two definitions of itself in the first place.
    """
    assert not rs.WORKERS_DB.exists()
    assert rs.sweep_worker_runs(apply=True, now=time.time()) == (0, 0, "")
    assert not rs.WORKERS_DB.exists(), "the sweep must not create a queue database"

    other = rs.WORKERS_DB.parent / "other.db"
    sqlite3.connect(other).close()
    assert rs.sweep_worker_runs(apply=True, now=time.time(), db=other) == (0, 0, "")
    conn = sqlite3.connect(other)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "runs" not in tables, "the sweep must not create the runs table either"


def test_the_bare_invocation_prunes_the_store_it_resolves(tmp_path):
    """#1018 clauses 1-3 across the boundary the weekly job actually runs on.

    Task #79 invokes `python3 retention-sweep.py --apply` under cron — an
    interpreter with no project venv, so the `workers.queue` import inside
    `_resolve_workers_db` fails and the fallback resolves the store as
    `DATA_ROOT / "workers.db"`. Every other test in this file runs in the venv, where
    that import *succeeds*, so none of them exercises the fallback that production
    actually takes; and the dry-run subprocess test only proves the line prints.

    This one runs the shipped script the way cron does — system `python3`, no venv, no
    pytest — against a data root of its own, and asserts the whole claim: the store
    line names its `>30d` horizon in both modes, `--apply` deletes the 31-day-old row
    and keeps the 29-day one, and nothing printed a path outside this test's tree.
    """
    py3 = shutil.which("python3")
    if not py3:
        pytest.skip("no bare python3 to prove the no-venv path against")
    root = tmp_path / "data"
    root.mkdir()
    db = root / "workers.db"
    _seed_runs_db(db)

    env = dict(os.environ)
    env["LLOYD_DATA"] = str(root)
    # An empty vault root: `AUTONOMY_TASKS_DIR` is derived from it, and the apply pass
    # must not reach the live board's task bodies while this test is proving a store.
    env["LLOYD_VAULT_ROOT"] = str(tmp_path / "no-vault")

    dry = subprocess.run([py3, str(_SCRIPT)], capture_output=True, text=True,
                         env=env, cwd="/", timeout=120)
    assert dry.returncode == 0, dry.stderr[-800:]
    dry_line = _store_line(dry.stdout, "workers.db")
    assert ">30d" in dry_line, (
        f"the bare interpreter reports the store without its horizon, so an operator "
        f"approving `--apply` cannot see how far back it reaches: {dry_line}")
    assert "1 deleted" in dry_line, (
        "dry-run must count exactly the one row past the horizon, because that number "
        f"is what the operator approves: {dry_line}")
    assert _run_ids(db) == {"run-past", "run-inside", "run-today"}, "dry run deleted rows"

    applied = subprocess.run([py3, str(_SCRIPT), "--apply"], capture_output=True,
                             text=True, env=env, cwd="/", timeout=120)
    assert applied.returncode == 0, applied.stderr[-800:]
    assert "Traceback" not in applied.stdout + applied.stderr
    assert _store_line(applied.stdout, "workers.db") == dry_line, (
        "the line an operator approved in dry-run must be the line apply prints")
    assert _run_ids(db) == {"run-inside", "run-today"}, (
        "the store the venv tests cover is not the store cron prunes: with no venv the "
        "`workers.queue` import fails, and the fallback resolved somewhere other than "
        "the data root this run was given")


def test_the_guard_names_any_unredirected_path_constant(rs):
    """#1018 clause 3, third half: the fixture's fail-loud guard catches any path.

    Before #1018 the guard looked only at names ending `_DIR`, so a `WORKERS_DB`
    that was never redirected would have been caught by nothing — and the
    consequence is not a failed assertion but a `DELETE` against the live queue,
    which is the one store a user's Background tab reads. A suffix list is not an
    answer either: it is only as wide as the last name somebody thought of. So the
    shipped rule is "every module-level `Path` that is not exempted by name", and
    this proves it by naming three constants no suffix rule would have seen — a
    `_FILE` variant of the name that already exists, a second store's database, and
    a root that is neither.

    The predicate is `_unredirected_destructive_paths`, the same function the fixture
    calls, so this exercises the shipped rule rather than a paraphrase of it.
    """
    tmp = rs.WORKERS_DB.parent

    # The shipped rule, applied to the module as the fixture left it: nothing
    # deletable is still pointing outside tmp_path.
    assert _unredirected_destructive_paths(rs, tmp) == []

    # The same rule against a module carrying the shapes this round found and the
    # shapes the next store might take, none of them redirected.
    class Unredirected:
        SESSIONS_DIR = rs.DATA_ROOT / "sessions"
        WORKERS_DB = rs.DATA_ROOT / "workers.db"
        WORKERS_DB_FILE = rs.DATA_ROOT / "workers.db"
        QUEUE_DB = rs.DATA_ROOT / "queue.db"
        SPILL_ROOT = rs.DATA_ROOT / "spill"
        SPILL_DIR_SUFFIX = ".tool-results"  # not a Path; never a candidate

    found = _unredirected_destructive_paths(Unredirected, tmp)
    assert found == ["QUEUE_DB", "SESSIONS_DIR", "SPILL_ROOT",
                     "WORKERS_DB", "WORKERS_DB_FILE"], (
        f"the guard reaches only {found} — a path constant still slips past it")

    # And the exemption is a written-down decision, not a blind spot: an exempted
    # name stops being reported, which is what `DATA_ROOT` and the script's own tree
    # root are for.
    assert _unredirected_destructive_paths(Unredirected, tmp,
                                          exempt=frozenset({"WORKERS_DB"})) == [
        "QUEUE_DB", "SESSIONS_DIR", "SPILL_ROOT", "WORKERS_DB_FILE"]


def test_the_skill_table_names_the_constant_and_split_the_code_uses(rs):
    """Clause 5, first half: `skills/retention-sweep/SKILL.md` agrees with the script.

    The skill is the operator's procedure for the weekly task, and its table already
    promised this prune with a `SKIPPED` form and a `WORKER_RUN_MAX_AGE_DAYS`
    constant while no code defined either — a procedure describing a rung that does
    not exist is worse than one that omits it, because the operator reads `0
    deleted` and believes the bound held. This pins the agreement in both
    directions: the name and age the code uses appear in the row, and the row for
    the spill store names the suffix and horizon the code uses.
    """
    skill = rs.vault_root() / "skills" / "retention-sweep" / "SKILL.md"
    if not skill.is_file():
        pytest.skip(f"the vault skill is not reachable from here: {skill}")

    rows = {ln.split("|")[1].strip(): ln
            for ln in skill.read_text(encoding="utf-8").splitlines()
            if ln.startswith("| ") and ln.count("|") >= 3}

    # Matched on `workers.db`, not on "runs": `autonomy-runs/<id>/run_*.md` is a
    # different store, on a different horizon, in the same table.
    runs_row = next((ln for store, ln in rows.items() if "workers.db" in store), None)
    assert runs_row is not None, f"no `runs` row in the skill's store table: {list(rows)}"
    assert "WORKER_RUN_MAX_AGE_DAYS" in runs_row, (
        "the skill names a constant the script does not define")
    assert f">{rs.WORKER_RUN_MAX_AGE_DAYS}d" in runs_row, runs_row
    assert "SKIPPED" in runs_row, runs_row

    spill_row = next((ln for store, ln in rows.items() if "tool-results" in store), None)
    assert spill_row is not None, (
        f"the skill omits the spill store the script sweeps: {list(rows)}")
    assert "SPILL_MAX_AGE_DAYS" in spill_row, spill_row
    assert f">{rs.SPILL_MAX_AGE_DAYS}d" in spill_row, spill_row


# ---------------------------------------------------------------------------
# The seventh rung: the activity logs in the vault's `autonomy/*.md` (#845).
#
# The cap kept the last ACTIVITY_LOG_MAX_ENTRIES entry bullets and reported a
# healthy prune, but it built that list from `- ` bullets only, so a line the
# writer had leaked into the file was in neither set: not an entry the cap could
# count, not a line any branch removed. `autonomy/24-data-pipeline.md` carried 359
# of them through every weekly sweep of that exact file while its own marker
# counted 6,841 entries pruned. So the rung now removes both shapes, and counts
# them apart, because a report of "6,841 entries pruned" about a file that shed
# junk points at failure loops that never happened.
# ---------------------------------------------------------------------------

#: A non-entry line of exactly the kind a multi-line note leaks: continuation text
#: at column 0, so not a bullet. This is the string the live corpus is full of.
BARE_JUNK = "Error output: Check stderr output for details"
#: The other leak shape: a nested bullet, which is a bullet under Markdown's rules
#: and under `lstrip()` — the reading that made the junk unremovable.
INDENTED_JUNK = " - 2026-09-20T07:55Z: nested continuation of one note"


def _entry(n: int) -> str:
    """Entry `n` of a synthetic log, shaped as `autonomy.append_activity_line` writes it.

    Stamped one minute apart and strictly increasing, so "the window kept the
    newest entries" is a check rather than a coincidence of duplicated text, and so
    no two seeds are equal and the collapse at the append site cannot fire inside
    these files.
    """
    hh, mm = divmod(n, 60)
    stamp = f"2026-09-20T{hh:02d}:{mm:02d}:00Z"
    return f"- {stamp}: Run run_24_20260920_{hh:02d}{mm:02d}00 — success (61s)"


def _activity_task(entries: int, *, bare_after: int = 0, bare_between: int = 0,
                   indented_between: int = 0, prior_marker: str = "") -> str:
    """A task body: `## Activity Log` with `entries` bullets, interleaved with the
    junk a note carrying newlines leaves behind.

    `prior_marker` is the line an earlier sweep wrote. Its counts are history the
    new marker carries forward, not something this sweep re-derives.
    """
    head = ["---", "id: 24", "name: Data pipeline", "status: up_next", "---", "",
            "# Data pipeline", "",
            "Overview prose, above the heading, that no sweep may touch.", "",
            "## Activity Log", ""]
    if prior_marker:
        head.append(prior_marker)
    lines = []
    for n in range(entries):
        lines.append(_entry(n))
        lines.extend([BARE_JUNK] * bare_between)
        lines.extend([INDENTED_JUNK] * indented_between)
    lines.extend([BARE_JUNK] * bare_after)
    return "\n".join(head + lines) + "\n"


def _activity_region(path: Path) -> list[str]:
    """Every non-blank line after `## Activity Log`, verbatim.

    The same reading the fleet check uses (`awk NR>heading`, drop blank, drop
    `/^- /`), because that expression is the one that has to fall to zero across
    the live vault — a test that read the file by some other rule could pass on a
    shape the real sweep leaves standing.
    """
    lines = path.read_text(encoding="utf-8").split("\n")
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().lower() == "## activity log")
    return [ln for ln in lines[start + 1:] if ln.strip()]


def _seed(rs, body: str, name: str = "24-data-pipeline.md") -> Path:
    path = rs.AUTONOMY_TASKS_DIR / name
    path.write_text(body, encoding="utf-8")
    return path


def test_a_sweep_removes_activity_lines_the_entry_cap_cannot_count(rs):
    """Clause 1: one `--apply` sweep removes every non-blank line under the heading
    that is neither an entry bullet nor the prune marker.

    The seed is deliberately *under* the cap — 3 entries, 7 junk lines — because
    that is where the old rung failed loudest: it computed `excess` from entry
    bullets, found none, and `continue`d before reaching anything that removed a
    line, so a file whose only defect was junk was counted clean and left standing.
    Junk has to go whether or not there is also something to prune.
    """
    path = _seed(rs, _activity_task(3, bare_after=4, indented_between=1))
    seeded = _activity_region(path)
    assert len(seeded) == 3 + 4 + 3, f"the seed lost its junk: {len(seeded)}"
    assert len([ln for ln in seeded if not ln.startswith("- ")]) == 7

    files, pruned, dropped = rs.sweep_activity_logs(apply=True)

    assert files == 1, "a file with junk and nothing to prune was skipped"
    after = _activity_region(path)
    assert [ln for ln in after if not ln.startswith("- ")] == [], (
        f"non-entry lines survived the sweep: "
        f"{[ln for ln in after if not ln.startswith('- ')]}")
    # Removing the leak must not remove the log: every real entry survives, in
    # order, and the prose above the heading was never in scope.
    assert [ln for ln in after if ln.startswith("- ") and not ln.startswith("- …")] \
        == [ln for ln in seeded if ln.startswith("- ")]
    assert "Overview prose, above the heading" in path.read_text(encoding="utf-8")
    # The two counts stay apart, and here nothing was pruned: reporting 7
    # "entries" for a file that ran 3 times is the misreport the marker carries.
    assert pruned == 0, f"junk counted as pruned entries: {pruned}"
    assert dropped == 7, f"non-entry lines removed: {dropped}"


def test_an_activity_log_cannot_exceed_its_cap_by_lines_the_cap_ignores(rs):
    """Clause 2: after the sweep, the non-blank lines following the heading are at
    most `ACTIVITY_LOG_MAX_ENTRIES + 1` — the retained entries plus one marker.

    Seeded over the cap *and* seeded with junk, which is the state that made the
    bound a lie: 200 retained entries plus 260 leaked lines is more than twice
    `ACTIVITY_LOG_MAX_ENTRIES`, on a file whose marker said "keeping last 200".
    """
    limit = rs.ACTIVITY_LOG_MAX_ENTRIES
    seeded_junk = (limit + 40) + 20
    path = _seed(rs, _activity_task(limit + 40, bare_after=20, bare_between=1))
    before = _activity_region(path)
    assert len(before) == (limit + 40) + seeded_junk
    assert len(before) > limit + 1, f"the seed is not over the bound: {len(before)}"

    files, pruned, dropped = rs.sweep_activity_logs(apply=True)

    after = _activity_region(path)
    assert files == 1
    assert len(after) <= limit + 1, (
        f"{len(after)} non-blank lines under the heading against a declared bound "
        f"of {limit} entries plus one marker")
    assert pruned == 40, f"entries past the cap: {pruned}"
    assert dropped == seeded_junk, f"non-entry lines removed: {dropped}"
    # And the window is the NEWEST entries: keeping the first 200 would satisfy
    # the size bound while throwing away the point of a rolling log.
    kept = [ln for ln in after if ln.startswith("- ") and not ln.startswith("- …")]
    assert kept == [_entry(n) for n in range(40, limit + 40)], kept[:3]
    assert kept[-1] == _entry(limit + 39)


def test_the_prune_marker_counts_pruned_entries_and_removed_lines_apart(rs):
    """The marker must not report junk as runs, and it stays one line.

    Guidance in the acceptance, pinned anyway, because the marker is the number a
    human reads and it is what made the old report unreadable. The marker carries
    the lifetime totals — what the cap has ever removed, and what has ever been
    recognised as leaked — which is why the fold must round-trip through the suffix
    form rather than drop it and restart at this run's number. The return carries
    what *this* run removed, which is what gets printed beside the other rungs and
    compared against last week's. One marker line, so the bound in clause 2 stays
    `MAX + 1`.
    """
    limit = rs.ACTIVITY_LOG_MAX_ENTRIES
    prior = (f"- … 6764 older entries pruned by retention sweep "
             f"(keeping last {limit}); non-entry lines removed: 12")
    path = _seed(rs, _activity_task(limit + 10, bare_after=5, prior_marker=prior))

    files, pruned, dropped = rs.sweep_activity_logs(apply=True)

    markers = [ln for ln in _activity_region(path) if ln.startswith("- …")]
    assert files == 1
    assert len(markers) == 1, f"markers were not folded into one: {markers}"
    assert markers[0] == (
        f"- … {6764 + 10} older entries pruned by retention sweep "
        f"(keeping last {limit}); non-entry lines removed: {12 + 5}"), markers[0]
    assert (pruned, dropped) == (10, 5), (
        f"the rung must report what this run removed, not the folded history: "
        f"{(pruned, dropped)}")


def test_the_activity_rung_reports_both_numbers_in_dry_run_without_writing(rs):
    """Dry run is what an operator approves `--apply` from, on both halves.

    The bound and the junk rule are new behaviour on a job the operator has run
    weekly for months, so the part that must not change is the promise: the same
    numbers, the file untouched, both shapes named — a dry run that printed only
    the entry count would leave the junk's size unknown before it was deleted.
    """
    path = _seed(rs, _activity_task(3, bare_after=4))

    files, pruned, dropped = rs.sweep_activity_logs(apply=False)

    assert (files, pruned, dropped) == (1, 0, 4), "the dry run counted something else"
    assert path.read_text(encoding="utf-8") == (
        _activity_task(3, bare_after=4)), "a dry run wrote the file"


def test_a_section_below_the_activity_log_is_not_swept(rs):
    """The region the junk rule reads is bounded at the next heading.

    Every live task file happens to end with its Activity Log, so "remove to
    end of file" reads as harmless today. It is not: a file that grows one
    section below its log would have that section's prose deleted as leaked note
    text, by a rung whose stated job is to remove only lines the cap cannot count.
    """
    body = _activity_task(2, bare_after=3) + (
        "\n## Notes\n\nA paragraph of real notes below the log.\n\n"
        "- a bullet in that section\n")
    path = _seed(rs, body)

    files, pruned, dropped = rs.sweep_activity_logs(apply=True)

    text = path.read_text(encoding="utf-8")
    assert files == 1
    assert dropped == 3, f"the sweep reached past the next heading: {dropped}"
    assert "A paragraph of real notes below the log." in text
    assert "- a bullet in that section" in text
    assert BARE_JUNK not in text
