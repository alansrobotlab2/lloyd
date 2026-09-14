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


def _write_session(dir: Path, name: str, last_active_days: float) -> Path:
    from datetime import datetime, timedelta
    ts = (datetime.now() - timedelta(days=last_active_days)).isoformat()
    p = dir / f"{name}.json"
    p.write_text(json.dumps({
        "session_id": name,
        "last_active": ts,
        "messages": [{"role": "user", "content": "hi"}],
    }, indent=2))
    return p


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
